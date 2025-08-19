// MongoDB Initialization Script for MasterX Production

// Switch to the masterx database
db = db.getSiblingDB('masterx');

// Create application user
db.createUser({
  user: 'masterx_app',
  pwd: 'masterx_app_password',
  roles: [
    {
      role: 'readWrite',
      db: 'masterx'
    }
  ]
});

// Create collections with indexes
db.createCollection('chat_messages');
db.createCollection('chat_sessions');
db.createCollection('uploaded_files');
db.createCollection('user_sessions');
db.createCollection('performance_metrics');

// Create indexes for performance
db.chat_messages.createIndex({ "session_id": 1, "timestamp": -1 });
db.chat_messages.createIndex({ "sender": 1, "timestamp": -1 });
db.chat_sessions.createIndex({ "session_id": 1 });
db.chat_sessions.createIndex({ "user_id": 1, "created_at": -1 });
db.uploaded_files.createIndex({ "file_id": 1 });
db.uploaded_files.createIndex({ "upload_time": -1 });
db.user_sessions.createIndex({ "session_id": 1 });
db.performance_metrics.createIndex({ "timestamp": -1 });

print('MasterX database initialized successfully');
