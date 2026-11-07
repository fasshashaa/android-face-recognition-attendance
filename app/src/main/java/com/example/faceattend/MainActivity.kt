package com.example.faceattend

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.Rect
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCase
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.faceattend.ui.theme.FaceAttendTheme
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executor
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.sqrt

// ======================================================================
// KONFIGURASI
// ======================================================================
private const val MATCH_THRESHOLD = 1.0f   // jarak Euclidean (embedding ter-normalisasi L2)
private const val MIN_MARGIN = 0.15f       // selisih minimal kandidat terbaik vs kedua
private const val SAMPLE_TARGET = 5
private const val SAMPLE_MIN = 3
private const val TAG = "FaceAttend"

// --- Liveness (anti-foto) ---
private const val CHALLENGE_TIMEOUT_MS = 8_000L   // batas waktu tiap tantangan
private const val ATTENDANCE_COOLDOWN_MS = 30_000L
private const val YAW_TURN_DEG = 20f              // sudut toleh yang dianggap "menoleh"
private const val YAW_CENTER_DEG = 12f            // wajah dianggap menghadap depan
// Arah toleh: jika "toleh kanan" ternyata terbaca kiri di perangkatmu, ubah menjadi 1f
private const val YAW_RIGHT_SIGN = -1f

private object FaceNetHolder {
    @Volatile private var instance: FaceNetInterpreter? = null
    fun get(context: Context): FaceNetInterpreter =
        instance ?: synchronized(this) {
            instance ?: FaceNetInterpreter(context.applicationContext).also { instance = it }
        }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            FaceAttendTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    CameraPermissionGate {
                        MainAppNavigation()
                    }
                }
            }
        }
    }
}

@Composable
fun CameraPermissionGate(content: @Composable () -> Unit) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }
    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted = it }

    LaunchedEffect(Unit) {
        if (!granted) launcher.launch(Manifest.permission.CAMERA)
    }

    if (granted) {
        content()
    } else {
        Column(
            modifier = Modifier.fillMaxSize().padding(24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text("Izin kamera diperlukan untuk absensi wajah.")
            Spacer(modifier = Modifier.height(12.dp))
            Button(onClick = { launcher.launch(Manifest.permission.CAMERA) }) {
                Text("Beri Izin Kamera")
            }
        }
    }
}

@Composable
fun MainAppNavigation() {
    val navController = rememberNavController()

    NavHost(navController = navController, startDestination = "home") {
        composable("home") {
            HomeScreen(
                onNavigateToUserManagement = { navController.navigate("users") },
                onNavigateToLogs = { navController.navigate("logs") }
            )
        }
        composable("users") {
            UserManagementScreen(onBack = { navController.popBackStack() })
        }
        composable("logs") {
            AttendanceLogScreen(onBack = { navController.popBackStack() })
        }
    }
}

// ======================================================================
// HELPER EMBEDDING
// ======================================================================
fun l2Normalize(v: FloatArray): FloatArray {
    var sum = 0f
    for (x in v) sum += x * x
    val norm = sqrt(sum).let { if (it < 1e-6f) 1f else it }
    return FloatArray(v.size) { v[it] / norm }
}

fun euclideanDistance(a: FloatArray, b: FloatArray): Float {
    if (a.size != b.size) return Float.MAX_VALUE
    var s = 0f
    for (i in a.indices) {
        val d = a[i] - b[i]
        s += d * d
    }
    return sqrt(s)
}

// ======================================================================
// DATA FRAME WAJAH (crop + sinyal liveness dari ML Kit)
// ======================================================================
class FaceFrame(
    val bitmap: Bitmap,
    val leftEyeOpen: Float?,
    val rightEyeOpen: Float?,
    val smile: Float?,
    val yaw: Float,
    val trackingId: Int?
)

// ======================================================================
// LIVENESS: tantangan acak yang harus dilakukan wajah ASLI.
// Foto statis tidak bisa berkedip / berubah ekspresi / menoleh di depan kamera.
// Setiap tantangan mensyaratkan PERUBAHAN keadaan (mis. mata terbuka -> tertutup
// -> terbuka), bukan sekadar nilai sesaat.
// ======================================================================
private enum class Challenge(val prompt: String) {
    BLINK("Berkedip 1x"),
    TURN_RIGHT("Toleh ke kanan"),
    TURN_LEFT("Toleh ke kiri"),
    SMILE("Tersenyum")
}

private class LivenessSession(val challenges: List<Challenge>) {
    var index = 0
        private set
    private var phase = 0
    private var startedAt = SystemClock.elapsedRealtime()

    val done: Boolean get() = index >= challenges.size
    val current: Challenge? get() = challenges.getOrNull(index)
    val timedOut: Boolean
        get() = SystemClock.elapsedRealtime() - startedAt > CHALLENGE_TIMEOUT_MS

    fun update(f: FaceFrame) {
        val c = current ?: return

        val completed: Boolean = when (c) {
            Challenge.BLINK -> {
                val l = f.leftEyeOpen
                val r = f.rightEyeOpen
                if (l == null || r == null) {
                    false
                } else {
                    val open = l > 0.7f && r > 0.7f
                    val closed = l < 0.3f && r < 0.3f
                    when (phase) {
                        0 -> { if (open) phase = 1; false }     // mata terbuka dulu
                        1 -> { if (closed) phase = 2; false }   // lalu tertutup
                        else -> open                            // lalu terbuka lagi = berkedip
                    }
                }
            }
            Challenge.SMILE -> {
                val s = f.smile
                if (s == null) {
                    false
                } else if (phase == 0) {
                    if (s < 0.3f) phase = 1                     // mulai dari wajah netral
                    false
                } else {
                    s > 0.8f
                }
            }
            Challenge.TURN_RIGHT, Challenge.TURN_LEFT -> {
                val dir = if (c == Challenge.TURN_RIGHT) YAW_RIGHT_SIGN else -YAW_RIGHT_SIGN
                if (phase == 0) {
                    if (abs(f.yaw) < YAW_CENTER_DEG) phase = 1  // mulai menghadap depan
                    false
                } else {
                    f.yaw * dir > YAW_TURN_DEG
                }
            }
        }

        if (completed) {
            index++
            phase = 0
            startedAt = SystemClock.elapsedRealtime()
        }
    }
}

private fun randomChallenges(): List<Challenge> {
    // Selalu ada KEDIP (paling sulit dipalsukan dengan foto) + 1 tantangan acak lain, urutan diacak
    val other = listOf(Challenge.TURN_LEFT, Challenge.TURN_RIGHT, Challenge.SMILE).random()
    return listOf(Challenge.BLINK, other).shuffled()
}

// ======================================================================
// ENGINE ABSENSI: identifikasi -> liveness -> catat hadir.
// Dipanggil dari thread analyzer kamera (satu thread), jadi tidak perlu lock.
// ======================================================================
private class AttendanceEngine(
    private val faceNet: FaceNetInterpreter,
    private val users: () -> List<Pair<UserEntity, FloatArray>>,
    private val onMessage: (String) -> Unit,
    private val onRecord: (UserEntity) -> Unit
) {
    private val lastAttendance = mutableMapOf<String, Long>()
    private var session: LivenessSession? = null
    private var candidate: UserEntity? = null
    private var trackingId: Int? = null
    private var lastFaceAt = 0L
    private var cooldownUntil = 0L
    private var mismatch = 0

    private fun reset(message: String, cooldownMs: Long = 0L) {
        session = null
        candidate = null
        trackingId = null
        mismatch = 0
        cooldownUntil = SystemClock.elapsedRealtime() + cooldownMs
        onMessage(message)
    }

    private fun prompt(s: LivenessSession) =
        "🔒 Verifikasi ${s.index + 1}/${s.challenges.size}: ${s.current?.prompt}"

    fun onFrame(frame: FaceFrame?) {
        val now = SystemClock.elapsedRealtime()
        if (now < cooldownUntil) return

        if (frame == null) {
            if (session != null) {
                if (now - lastFaceAt > 1500) reset("Wajah hilang dari kamera, ulangi.", 1000)
            } else {
                onMessage("Arahkan wajah ke kamera...")
            }
            return
        }
        lastFaceAt = now

        val saved = users()
        if (saved.isEmpty()) {
            onMessage("⚠️ Belum ada user terdaftar!")
            return
        }

        val emb = l2Normalize(faceNet.getFaceEmbedding(frame.bitmap))

        var best: UserEntity? = null
        var bestD = Float.MAX_VALUE
        var secondD = Float.MAX_VALUE
        for ((u, e) in saved) {
            val d = euclideanDistance(emb, e)
            if (d < bestD) {
                secondD = bestD
                bestD = d
                best = u
            } else if (d < secondD) {
                secondD = d
            }
        }

        val s = session

        // ---------- Tahap 1: kenali dulu siapa orangnya ----------
        if (s == null) {
            val passThreshold = best != null && bestD < MATCH_THRESHOLD
            val passMargin = secondD == Float.MAX_VALUE || (secondD - bestD) >= MIN_MARGIN
            if (!passThreshold) {
                onMessage("❌ Wajah tidak dikenali (Dist: %.2f)".format(bestD))
                return
            }
            if (!passMargin) {
                onMessage("⚠️ Wajah ambigu, coba lagi (%.2f vs %.2f)".format(bestD, secondD))
                return
            }

            val user = best!!
            val last = lastAttendance[user.employeeId]
            if (last != null && now - last < ATTENDANCE_COOLDOWN_MS) {
                onMessage("ℹ️ ${user.name} sudah absen.")
                cooldownUntil = now + 2000
                return
            }

            candidate = user
            trackingId = frame.trackingId
            mismatch = 0
            val newSession = LivenessSession(randomChallenges())
            session = newSession
            onMessage(prompt(newSession))
            return
        }

        // ---------- Tahap 2: liveness, wajah harus tetap orang yang sama ----------
        val cand = candidate!!
        val sameFace = best?.employeeId == cand.employeeId && bestD < MATCH_THRESHOLD + 0.2f
        val tid = trackingId
        val sameTrack = tid == null || frame.trackingId == null || frame.trackingId == tid

        if (sameFace && sameTrack) {
            mismatch = 0
        } else {
            mismatch++
            if (mismatch > 3) {
                reset("❌ Wajah berganti / tidak cocok. Ulangi.", 2000)
            }
            return
        }

        s.update(frame)

        when {
            s.done -> {
                lastAttendance[cand.employeeId] = now
                onRecord(cand)
                reset("✅ Hadir: ${cand.name} (Dist: %.2f)".format(bestD), 3000)
            }
            s.timedOut -> reset("⌛ Waktu verifikasi habis. Ulangi.", 1500)
            else -> onMessage(prompt(s))
        }
    }
}

// ======================================================================
// 1. HALAMAN UTAMA
// ======================================================================
@Composable
fun HomeScreen(onNavigateToUserManagement: () -> Unit, onNavigateToLogs: () -> Unit) {
    val context = LocalContext.current
    val db = AppDatabase.getDatabase(context)
    val userDao = db.attendanceDao()

    val userList by userDao.getAllUsers().collectAsState(initial = emptyList())
    val faceNetResult = remember { runCatching { FaceNetHolder.get(context) } }
    val faceNet = faceNetResult.getOrNull()

    val savedEmbeddings = remember(userList) {
        userList
            .filter { it.faceEmbedding.isNotBlank() }
            .map { it to l2Normalize(FaceNetInterpreter.stringToFloatArray(it.faceEmbedding)) }
    }
    val savedState = rememberUpdatedState(savedEmbeddings)

    var message by remember { mutableStateOf("Arahkan wajah ke kamera...") }

    val engine = remember(faceNet) {
        faceNet?.let {
            AttendanceEngine(
                faceNet = it,
                users = { savedState.value },
                onMessage = { m -> message = m },
                onRecord = { u -> recordAttendance(context, u.name, u.employeeId) }
            )
        }
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Button(onClick = onNavigateToUserManagement) { Text("👥 Kelola User") }
            Button(onClick = onNavigateToLogs) { Text("📋 Lihat Log Absen") }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Box(
            modifier = Modifier.weight(1f).fillMaxWidth().background(Color.Black),
            contentAlignment = Alignment.BottomCenter
        ) {
            // Interval 120 ms agar kedipan (±150-300 ms) tidak terlewat
            CameraPreview(modifier = Modifier.fillMaxSize(), minIntervalMs = 120) { frame ->
                if (engine == null) {
                    message = "❌ Model gagal dimuat: ${faceNetResult.exceptionOrNull()?.message}"
                } else {
                    try {
                        engine.onFrame(frame)
                    } catch (e: Exception) {
                        Log.e(TAG, "Gagal memproses frame", e)
                        message = "Gagal memproses wajah: ${e.message}"
                    }
                }
            }

            Text(
                text = message,
                color = Color.White,
                modifier = Modifier
                    .padding(16.dp)
                    .background(Color.Black.copy(alpha = 0.6f), shape = RoundedCornerShape(8.dp))
                    .padding(12.dp),
                style = MaterialTheme.typography.bodyLarge
            )
        }
    }
}

// ======================================================================
// ANALYZER KAMERA
// ======================================================================
private class FaceAnalyzer(
    private val detector: FaceDetector,
    private val executor: Executor,
    private val minIntervalMs: Long,
    private val onFace: (FaceFrame?) -> Unit
) : ImageAnalysis.Analyzer {

    @Volatile private var busy = false
    private var lastTs = 0L

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    override fun analyze(imageProxy: ImageProxy) {
        val now = SystemClock.elapsedRealtime()
        val media = imageProxy.image
        if (busy || now - lastTs < minIntervalMs || media == null) {
            imageProxy.close()
            return
        }
        busy = true
        lastTs = now

        val rotation = imageProxy.imageInfo.rotationDegrees
        try {
            val input = InputImage.fromMediaImage(media, rotation)
            // Semua listener dijalankan di executor analyzer (background), bukan main thread
            detector.process(input)
                .addOnSuccessListener(executor) { faces ->
                    val face: Face? = faces.maxByOrNull {
                        it.boundingBox.width() * it.boundingBox.height()
                    }
                    if (face == null) {
                        onFace(null)
                        return@addOnSuccessListener
                    }
                    try {
                        val raw = imageProxy.toBitmap()
                        val upright = if (rotation != 0) {
                            val m = Matrix().apply { postRotate(rotation.toFloat()) }
                            Bitmap.createBitmap(raw, 0, 0, raw.width, raw.height, m, true)
                        } else raw
                        val crop = cropFaceSquare(upright, face.boundingBox)
                        if (crop == null) {
                            onFace(null)
                        } else {
                            onFace(
                                FaceFrame(
                                    bitmap = crop,
                                    leftEyeOpen = face.leftEyeOpenProbability,
                                    rightEyeOpen = face.rightEyeOpenProbability,
                                    smile = face.smilingProbability,
                                    yaw = face.headEulerAngleY,
                                    trackingId = face.trackingId
                                )
                            )
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Gagal membuat bitmap wajah", e)
                        onFace(null)
                    }
                }
                .addOnFailureListener(executor) { onFace(null) }
                .addOnCompleteListener(executor) {
                    imageProxy.close()
                    busy = false
                }
        } catch (e: Exception) {
            Log.e(TAG, "Analyzer error", e)
            imageProxy.close()
            busy = false
        }
    }
}

private fun cropFaceSquare(src: Bitmap, box: Rect): Bitmap? {
    val side = (maxOf(box.width(), box.height()) * 1.2f).toInt()
        .coerceAtMost(minOf(src.width, src.height))
    if (side < 60) return null

    val left = (box.centerX() - side / 2).coerceIn(0, src.width - side)
    val top = (box.centerY() - side / 2).coerceIn(0, src.height - side)

    val cropped = Bitmap.createBitmap(src, left, top, side, side)
    return Bitmap.createScaledBitmap(cropped, 112, 112, true)
}

@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    minIntervalMs: Long = 300,
    onFrame: (FaceFrame?) -> Unit
) {
    val context = LocalContext.current
    val currentCallback by rememberUpdatedState(onFrame)

    val providerFuture = remember { ProcessCameraProvider.getInstance(context) }
    val executor = remember { Executors.newSingleThreadExecutor() }
    val boundUseCases = remember { mutableListOf<UseCase>() }
    val detector = remember {
        FaceDetection.getClient(
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                // Dibutuhkan untuk liveness: probabilitas mata terbuka & senyum
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                // Dibutuhkan agar wajah yang sama bisa dilacak selama tantangan
                .enableTracking()
                .build()
        )
    }

    DisposableEffect(Unit) {
        onDispose {
            boundUseCases.filterIsInstance<ImageAnalysis>().forEach { runCatching { it.clearAnalyzer() } }
            runCatching {
                if (boundUseCases.isNotEmpty()) {
                    providerFuture.get().unbind(*boundUseCases.toTypedArray())
                }
            }
            boundUseCases.clear()
            detector.close()
            executor.shutdown()
        }
    }

    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            val previewView = PreviewView(ctx)

            providerFuture.addListener({
                val cameraProvider = providerFuture.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        it.setAnalyzer(
                            executor,
                            FaceAnalyzer(detector, executor, minIntervalMs) { f -> currentCallback(f) }
                        )
                    }

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        ctx as androidx.lifecycle.LifecycleOwner,
                        CameraSelector.DEFAULT_FRONT_CAMERA,
                        preview,
                        imageAnalysis
                    )
                    boundUseCases.clear()
                    boundUseCases.add(preview)
                    boundUseCases.add(imageAnalysis)
                } catch (exc: Exception) {
                    Log.e(TAG, "Gagal memasang kamera", exc)
                }
            }, ContextCompat.getMainExecutor(ctx))

            previewView
        }
    )
}

// ======================================================================
// 2. HALAMAN MANAJEMEN USER
// ======================================================================
@Composable
fun UserManagementScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val db = AppDatabase.getDatabase(context)
    val userDao = db.attendanceDao()

    val userList by userDao.getAllUsers().collectAsState(initial = emptyList())

    var nameInput by remember { mutableStateOf("") }
    var idInput by remember { mutableStateOf("") }
    var liveFace by remember { mutableStateOf<Bitmap?>(null) }
    val samples = remember { mutableStateListOf<FloatArray>() }
    var isCapturing by remember { mutableStateOf(false) }
    var statusMessage by remember { mutableStateOf("Isi nama & ID, lalu tekan Rekam Wajah.") }

    val scope = rememberCoroutineScope()
    val faceNetResult = remember { runCatching { FaceNetHolder.get(context) } }
    val faceNet = faceNetResult.getOrNull()

    LaunchedEffect(faceNet) {
        if (faceNet == null) {
            statusMessage = "❌ Model gagal dimuat: ${faceNetResult.exceptionOrNull()?.message}"
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Start) {
            Button(onClick = onBack) { Text("← Kembali") }
        }

        Spacer(modifier = Modifier.height(8.dp))
        Text(text = "👥 Pendaftaran User & Wajah", style = MaterialTheme.typography.titleLarge)
        Spacer(modifier = Modifier.height(12.dp))

        OutlinedTextField(
            value = nameInput,
            onValueChange = { nameInput = it },
            label = { Text("Nama Lengkap") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(8.dp))
        OutlinedTextField(
            value = idInput,
            onValueChange = { idInput = it },
            label = { Text("NIP / ID Karyawan") },
            singleLine = true,
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(12.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(260.dp)
                .background(Color.Black)
        ) {
            CameraPreview(modifier = Modifier.fillMaxSize()) { frame -> liveFace = frame?.bitmap }
        }

        Spacer(modifier = Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            val preview = liveFace
            if (preview != null) {
                Image(
                    bitmap = preview.asImageBitmap(),
                    contentDescription = "Wajah terdeteksi",
                    modifier = Modifier.size(72.dp)
                )
            } else {
                Box(modifier = Modifier.size(72.dp).background(Color.LightGray))
            }
            Spacer(modifier = Modifier.width(12.dp))
            Text(
                text = if (preview != null) "Wajah terdeteksi ✅" else "Belum ada wajah terdeteksi",
                style = MaterialTheme.typography.bodyMedium
            )
        }

        Spacer(modifier = Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                enabled = !isCapturing && faceNet != null,
                onClick = {
                    if (faceNet == null) return@Button
                    scope.launch {
                        isCapturing = true
                        samples.clear()
                        val deadline = SystemClock.elapsedRealtime() + 20_000
                        var lastUsed: Bitmap? = null

                        while (samples.size < SAMPLE_TARGET &&
                            SystemClock.elapsedRealtime() < deadline
                        ) {
                            val face = liveFace
                            if (face == null) {
                                statusMessage = "Hadapkan wajah ke kamera..."
                                delay(200)
                                continue
                            }
                            if (face === lastUsed) {
                                delay(100)
                                continue
                            }
                            lastUsed = face
                            try {
                                val emb = withContext(Dispatchers.IO) {
                                    l2Normalize(faceNet.getFaceEmbedding(face))
                                }
                                samples.add(emb)
                                statusMessage = "Merekam sampel ${samples.size}/$SAMPLE_TARGET " +
                                        "— gerakkan kepala sedikit"
                            } catch (e: Exception) {
                                Log.e(TAG, "Gagal ekstrak embedding", e)
                                statusMessage = "❌ Gagal memproses wajah: ${e.message}"
                                break
                            }
                            delay(500)
                        }

                        isCapturing = false
                        if (samples.size >= SAMPLE_TARGET) {
                            statusMessage = "✅ $SAMPLE_TARGET sampel terekam. Tekan Simpan."
                        } else if (!statusMessage.startsWith("❌")) {
                            statusMessage = "⚠️ Hanya ${samples.size} sampel terekam. " +
                                    "Pastikan wajah terlihat jelas, lalu ulangi."
                        }
                    }
                },
                modifier = Modifier.weight(1f)
            ) {
                Text(if (isCapturing) "Merekam ${samples.size}/$SAMPLE_TARGET..." else "📸 Rekam Wajah")
            }

            OutlinedButton(
                enabled = !isCapturing,
                onClick = {
                    samples.clear()
                    statusMessage = "Sampel dihapus."
                },
                modifier = Modifier.weight(1f)
            ) { Text("🔄 Reset") }
        }

        Spacer(modifier = Modifier.height(6.dp))
        Text(text = statusMessage, style = MaterialTheme.typography.bodySmall, color = Color.Gray)
        Spacer(modifier = Modifier.height(12.dp))

        Button(
            enabled = !isCapturing && samples.size >= SAMPLE_MIN,
            onClick = {
                val inputName = nameInput.trim()
                val inputId = idInput.trim()

                if (inputName.isBlank() || inputId.isBlank()) {
                    statusMessage = "⚠️ Nama dan ID wajib diisi!"
                    return@Button
                }
                if (userList.any { it.employeeId == inputId }) {
                    statusMessage = "⚠️ ID $inputId sudah terdaftar. Gunakan ID lain."
                    return@Button
                }

                val snapshot = samples.toList()
                val existing = userList
                    .filter { it.faceEmbedding.isNotBlank() }
                    .map { it.name to l2Normalize(FaceNetInterpreter.stringToFloatArray(it.faceEmbedding)) }

                scope.launch(Dispatchers.IO) {
                    try {
                        val dim = snapshot[0].size
                        val mean = FloatArray(dim) { i ->
                            snapshot.sumOf { it[i].toDouble() }.toFloat() / snapshot.size
                        }
                        val finalEmbedding = l2Normalize(mean)

                        val closest = existing.minByOrNull { euclideanDistance(finalEmbedding, it.second) }
                        val closestInfo = closest?.let {
                            " Jarak ke ${it.first}: %.2f".format(euclideanDistance(finalEmbedding, it.second))
                        } ?: ""

                        userDao.insertUser(
                            UserEntity(
                                name = inputName,
                                employeeId = inputId,
                                imagePath = "face_path.jpg",
                                faceEmbedding = FaceNetInterpreter.floatArrayToString(finalEmbedding)
                            )
                        )

                        withContext(Dispatchers.Main) {
                            statusMessage = "🎉 Berhasil mendaftarkan $inputName!$closestInfo"
                            nameInput = ""
                            idInput = ""
                            samples.clear()
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Gagal menyimpan user", e)
                        withContext(Dispatchers.Main) {
                            statusMessage = "❌ Gagal menyimpan: ${e.message}"
                        }
                    }
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(text = "Simpan Data & Wajah User")
        }

        Spacer(modifier = Modifier.height(16.dp))
        Text(text = "Daftar User Terdaftar (${userList.size}):", style = MaterialTheme.typography.titleMedium)
        Spacer(modifier = Modifier.height(8.dp))

        userList.forEach { user ->
            Card(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                Column(modifier = Modifier.padding(12.dp)) {
                    Text(text = "Nama: ${user.name}", style = MaterialTheme.typography.bodyLarge)
                    Text(text = "ID: ${user.employeeId}", style = MaterialTheme.typography.bodyMedium)
                    Text(
                        text = "Embedding status: ${if (user.faceEmbedding.isNotBlank()) "✅ Tersimpan" else "❌ Kosong"}",
                        style = MaterialTheme.typography.bodySmall,
                        color = Color.Gray
                    )
                }
            }
        }
    }
}

// ======================================================================
// 3. HALAMAN LOG ABSEN
// ======================================================================
@Composable
fun AttendanceLogScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val db = AppDatabase.getDatabase(context)
    val userDao = db.attendanceDao()

    val logList by userDao.getAllLogs().collectAsState(initial = emptyList())

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
    ) {
        Button(onClick = onBack) { Text("← Kembali") }
        Spacer(modifier = Modifier.height(16.dp))

        Text(text = "📋 Rekapitulasi Kehadiran", style = MaterialTheme.typography.titleLarge)
        Spacer(modifier = Modifier.height(12.dp))

        LazyColumn(modifier = Modifier.fillMaxWidth().weight(1f)) {
            items(items = logList, key = { log -> log.id }) { log ->
                Card(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(text = "Nama: ${log.userName}", style = MaterialTheme.typography.bodyLarge)
                        Text(text = "ID: ${log.employeeId}", style = MaterialTheme.typography.bodyMedium)
                        Text(text = "Waktu: ${log.timestamp}", style = MaterialTheme.typography.bodySmall, color = Color.Gray)
                    }
                }
            }
        }
    }
}

fun recordAttendance(context: Context, name: String, id: String) {
    val db = AppDatabase.getDatabase(context)
    val dao = db.attendanceDao()
    val currentTime = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault())
        .format(java.util.Date())

    kotlinx.coroutines.CoroutineScope(Dispatchers.IO).launch {
        dao.insertLog(
            AttendanceLogEntity(
                userName = name,
                employeeId = id,
                timestamp = currentTime
            )
        )
    }
}