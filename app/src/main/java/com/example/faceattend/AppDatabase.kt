package com.example.faceattend

import android.content.Context
import androidx.room.*
import kotlinx.coroutines.flow.Flow

// ==========================================
// 1. ENTITY / TABEL USER
// ==========================================
@Entity(tableName = "users")
data class UserEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val name: String,
    val employeeId: String,
    val imagePath: String, // Path foto wajah yang didaftarkan
    val faceEmbedding: String = ""
)

// ==========================================
// 2. ENTITY / TABEL LOG ABSENSI
// ==========================================
@Entity(tableName = "attendance_logs")
data class AttendanceLogEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val userName: String,
    val employeeId: String,
    val timestamp: String // Format waktu saat absen
)

// ==========================================
// 3. DAO (Data Access Object) - Perintah SQL
// ==========================================
@Dao
interface AttendanceDao {
    @Query("SELECT * FROM users")
    fun getAllUsers(): Flow<List<UserEntity>>

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertUser(user: UserEntity)

    @Query("SELECT * FROM attendance_logs ORDER BY id DESC")
    fun getAllLogs(): Flow<List<AttendanceLogEntity>>

    @Insert
    suspend fun insertLog(log: AttendanceLogEntity)
}

// ==========================================
// 4. DATABASE INSTANCE (Harus Abstract Class)
// ==========================================
@Database(
    entities = [UserEntity::class, AttendanceLogEntity::class],
    version = 1,
    exportSchema = false
)
abstract class AppDatabase : RoomDatabase() {
    abstract fun attendanceDao(): AttendanceDao

    companion object {
        @Volatile
        private var INSTANCE: AppDatabase? = null

        fun getDatabase(context: Context): AppDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    AppDatabase::class.java,
                    "face_attend_database"
                )
                    .fallbackToDestructiveMigration()
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}