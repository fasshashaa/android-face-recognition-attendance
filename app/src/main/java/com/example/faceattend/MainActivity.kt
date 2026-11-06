package com.example.faceattend

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.faceattend.ui.theme.FaceAttendTheme
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.ui.viewinterop.AndroidView
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import androidx.compose.runtime.getValue
import androidx.compose.runtime.setValue
import androidx.compose.foundation.background
import androidx.compose.ui.graphics.Color
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.Alignment
import androidx.compose.ui.platform.LocalContext
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.LazyColumn


class MainActivity : ComponentActivity() {
    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (!isGranted) {
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Cek izin kamera saat app pertama kali nyala
        if (androidx.core.content.ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        setContent {
            FaceAttendTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainAppNavigation()
                }
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
                onNavigateToUsers = { navController.navigate("users") },
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

// 1. Halaman Utama (Kamera & Menu Navigasi)
@Composable
fun HomeScreen(onNavigateToUsers: () -> Unit, onNavigateToLogs: () -> Unit) {
    // State untuk mendeteksi status wajah secara real-time
    var isFaceDetected by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Text(text = "🤖 FaceAttend Kiosk", style = MaterialTheme.typography.headlineMedium)
        Spacer(modifier = Modifier.height(8.dp))
        Text(text = "Arahkan wajah ke kamera untuk melakukan absensi.")

        Spacer(modifier = Modifier.height(24.dp))

        // Kotak Preview Kamera + Indikator Wajah
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .height(300.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
        ) {
            Box(modifier = Modifier.fillMaxSize()) {
                // Memanggil Preview Kamera yang sudah dipasangi ML Kit Analyzer
                CameraPreview(
                    modifier = Modifier.fillMaxSize(),
                    onFaceDetected = { detected ->
                        isFaceDetected = detected
                    }
                )

                // Indikator Status di atas Kamera
                Box(
                    modifier = Modifier
                        .align(Alignment.TopCenter)
                        .padding(12.dp)
                        .background(
                            color = if (isFaceDetected) Color.Green.copy(alpha = 0.7f) else Color.Red.copy(alpha = 0.7f),
                            shape = RoundedCornerShape(8.dp)
                        )
                        .padding(horizontal = 12.dp, vertical = 6.dp)
                ) {
                    Text(
                        text = if (isFaceDetected) "✅ Wajah Terdeteksi" else "⚠️ Cari Wajah...",
                        color = Color.White,
                        style = MaterialTheme.typography.bodyMedium
                    )
                }
            }
        }
        Spacer(modifier = Modifier.height(24.dp))

        // CONTOH TOMBOL SIMULASI ABSEN MASUK OTOMATIS
        val context = LocalContext.current
        Button(
            onClick = {
                // Contoh mencatat absen untuk user pertama (atau bisa disesuaikan nanti)
                recordAttendance(context, name = "Fassha Fanny", id = "EMP001")
            },
            modifier = Modifier.fillMaxWidth(),
            colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF267B9A))
        ) {
            Text(text = "⚡ Simulasi Absen Masuk Sekarang", color = Color.White)
        }

        Spacer(modifier = Modifier.height(8.dp))
        Button(onClick = onNavigateToUsers, modifier = Modifier.fillMaxWidth()) {
            Text(text = "Kelola Data Wajah / User")
        }
        Spacer(modifier = Modifier.height(8.dp))
        OutlinedButton(onClick = onNavigateToLogs, modifier = Modifier.fillMaxWidth()) {
            Text(text = "Lihat Rekap Absensi")
        }
    }
}

@Composable
fun CameraPreview(
    modifier: Modifier = Modifier,
    onFaceDetected: (Boolean) -> Unit
) {
    val context = androidx.compose.ui.platform.LocalContext.current
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }

    // Konfigurasi ML Kit Face Detector
    val highAccuracyOpts = FaceDetectorOptions.Builder()
        .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
        .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
        .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
        .build()

    val detector = FaceDetection.getClient(highAccuracyOpts)

    AndroidView(
        modifier = modifier,
        factory = { ctx ->
            val previewView = PreviewView(ctx)
            val cameraExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()

            cameraProviderFuture.addListener({
                val cameraProvider = cameraProviderFuture.get()

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

                // Setup ImageAnalysis untuk menangkap frame per frame secara real-time
                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()

                imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                    processImageProxy(detector, imageProxy, onFaceDetected)
                }

                val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

                try {
                    cameraProvider.unbindAll()
                    cameraProvider.bindToLifecycle(
                        ctx as androidx.lifecycle.LifecycleOwner,
                        cameraSelector,
                        preview,
                        imageAnalysis // Pasang imageAnalysis ke lifecycle kamera
                    )
                } catch (exc: Exception) {
                    // Handle error
                }
            }, ContextCompat.getMainExecutor(ctx))

            previewView
        }
    )
}

// Fungsi bantu untuk memproses frame kamera ke ML Kit
@androidx.annotation.OptIn(androidx.camera.core.ExperimentalGetImage::class)
private fun processImageProxy(
    detector: com.google.mlkit.vision.face.FaceDetector,
    imageProxy: ImageProxy,
    onFaceDetected: (Boolean) -> Unit
) {
    val mediaImage = imageProxy.image
    if (mediaImage != null) {
        val image = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)

        detector.process(image)
            .addOnSuccessListener { faces ->
                // Jika faces tidak kosong, berarti ada wajah di depan kamera
                onFaceDetected(faces.isNotEmpty())
            }
            .addOnFailureListener {
                onFaceDetected(false)
            }
            .addOnCompleteListener {
                imageProxy.close() // Tutup proxy agar frame selanjutnya bisa diproses
            }
    } else {
        imageProxy.close()
    }
}

// 2. Halaman Manajemen User (Dummy sementara)
// 2. Halaman Manajemen User
@Composable
fun UserManagementScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val db = AppDatabase.getDatabase(context)
    val userDao = db.attendanceDao()

    val userList by userDao.getAllUsers().collectAsState(initial = emptyList())

    var nameInput by remember { mutableStateOf("") }
    var idInput by remember { mutableStateOf("") }
    val scope = rememberCoroutineScope()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp)
    ) {
        Button(onClick = onBack) { Text("← Kembali") }
        Spacer(modifier = Modifier.height(16.dp))

        Text(text = "👥 Pendaftaran User Baru", style = MaterialTheme.typography.titleLarge)
        Spacer(modifier = Modifier.height(12.dp))

        // Input Form Sederhana
        OutlinedTextField(
            value = nameInput,
            onValueChange = { nameInput = it },
            label = { Text("Nama Lengkap") },
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(8.dp))
        OutlinedTextField(
            value = idInput,
            onValueChange = { idInput = it },
            label = { Text("NIP / ID Karyawan") },
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(modifier = Modifier.height(12.dp))

        Button(
            onClick = {
                if (nameInput.isBlank() || idInput.isBlank()) return@Button

                scope.launch(Dispatchers.IO) {
                    userDao.insertUser(
                        UserEntity(name = nameInput, employeeId = idInput, imagePath = "dummy_path.jpg")
                    )
                }
                nameInput = ""
                idInput = ""
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text(text = "Simpan Data User")
        }

        Spacer(modifier = Modifier.height(24.dp))
        Text(text = "Daftar User Terdaftar:", style = MaterialTheme.typography.titleMedium)
        Spacer(modifier = Modifier.height(8.dp))

        // PERBAIKAN UTAMA DI SINI: Gunakan .weight(1f) agar LazyColumn mengambil sisa layar secara fleksibel
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f) // <-- KUNCI AGAR LIST MUNCUL DAN BISA SCROLL
        ) {
            items(
                items = userList,
                key = { user -> user.id }
            ) { user ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                ) {
                    Column(modifier = Modifier.padding(12.dp)) {
                        Text(text = "Nama: ${user.name}", style = MaterialTheme.typography.bodyLarge)
                        Text(text = "ID: ${user.employeeId}", style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
    }
}

// 3. Halaman Log Absen (Rekap Kehadiran)
@Composable
fun AttendanceLogScreen(onBack: () -> Unit) {
    val context = LocalContext.current
    val db = AppDatabase.getDatabase(context)
    val userDao = db.attendanceDao()

    // Mengambil rekap log absensi secara reaktif dari database menggunakan Flow
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

        // Menampilkan daftar log absensi menggunakan LazyColumn yang fleksibel
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .weight(1f)
        ) {
            items(
                items = logList,
                key = { log -> log.id }
            ) { log ->
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(vertical = 4.dp)
                ) {
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
// Contoh fungsi helper untuk menyimpan data absen ke database
fun recordAttendance(context: android.content.Context, name: String, id: String) {
    val db = AppDatabase.getDatabase(context)
    val dao = db.attendanceDao()
    val currentTime = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", java.util.Locale.getDefault()).format(java.util.Date())

    kotlinx.coroutines.CoroutineScope(kotlinx.coroutines.Dispatchers.IO).launch {
        dao.insertLog(
            AttendanceLogEntity(
                userName = name,
                employeeId = id,
                timestamp = currentTime
            )
        )
    }
}