from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
import os
import uuid
import mimetypes
from pathlib import Path
from typing import List, Optional
import aiofiles
import asyncio
from datetime import datetime

# File processing imports
import base64
from io import BytesIO

router = APIRouter()

# Configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {
    # Documents
    '.pdf', '.doc', '.docx', '.txt', '.md', '.rtf',
    # Images
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg',
    # Code files
    '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.json', '.xml', '.yaml', '.yml',
    '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift',
    # Data files
    '.csv', '.xlsx', '.xls', '.json', '.xml',
    # Archives
    '.zip', '.tar', '.gz'
}

def get_file_type(filename: str) -> str:
    """Determine file type based on extension"""
    ext = Path(filename).suffix.lower()
    
    if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']:
        return 'image'
    elif ext in ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf']:
        return 'document'
    elif ext in ['.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.json', '.xml', '.yaml', '.yml',
                 '.java', '.cpp', '.c', '.h', '.cs', '.php', '.rb', '.go', '.rs', '.swift']:
        return 'code'
    elif ext in ['.csv', '.xlsx', '.xls']:
        return 'data'
    elif ext in ['.zip', '.tar', '.gz']:
        return 'archive'
    else:
        return 'other'

async def process_text_file(file_path: Path) -> str:
    """Extract text content from text-based files"""
    try:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            return content[:10000]  # Limit to first 10k characters
    except UnicodeDecodeError:
        try:
            async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                content = await f.read()
                return content[:10000]
        except Exception as e:
            return f"Error reading file: {str(e)}"

async def process_image_file(file_path: Path) -> dict:
    """Process image files and extract basic info"""
    try:
        # For now, just return basic file info
        # In production, you'd use PIL or similar for image analysis
        stat = file_path.stat()
        return {
            "type": "image",
            "size": stat.st_size,
            "description": "Image file uploaded for analysis"
        }
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a single file"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"File type {file_ext} not allowed. Supported types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    safe_filename = f"{file_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_filename
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    # Process file based on type
    file_type = get_file_type(file.filename)
    processed_content = None
    
    try:
        if file_type in ['document', 'code'] or file_ext in ['.txt', '.md', '.json', '.xml', '.yaml', '.yml']:
            processed_content = await process_text_file(file_path)
        elif file_type == 'image':
            processed_content = await process_image_file(file_path)
        else:
            processed_content = f"File uploaded: {file.filename} ({file_type})"
    except Exception as e:
        processed_content = f"Error processing file: {str(e)}"
    
    # Return file info
    return {
        "file_id": file_id,
        "filename": file.filename,
        "file_type": file_type,
        "size": len(content),
        "upload_time": datetime.now().isoformat(),
        "processed_content": processed_content,
        "status": "uploaded"
    }

@router.post("/upload-multiple")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Upload and process multiple files"""
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Too many files. Maximum 10 files per upload.")
    
    results = []
    for file in files:
        try:
            result = await upload_file(file)
            results.append(result)
        except HTTPException as e:
            results.append({
                "filename": file.filename,
                "error": e.detail,
                "status": "failed"
            })
    
    return {
        "uploaded_files": results,
        "total_files": len(files),
        "successful_uploads": len([r for r in results if r.get("status") == "uploaded"])
    }

@router.get("/file/{file_id}")
async def get_file_info(file_id: str):
    """Get information about an uploaded file"""
    
    # Find file by ID
    for file_path in UPLOAD_DIR.glob(f"{file_id}_*"):
        if file_path.is_file():
            stat = file_path.stat()
            return {
                "file_id": file_id,
                "filename": file_path.name.split("_", 1)[1],  # Remove UUID prefix
                "size": stat.st_size,
                "upload_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "file_type": get_file_type(file_path.name),
                "status": "available"
            }
    
    raise HTTPException(status_code=404, detail="File not found")

@router.delete("/file/{file_id}")
async def delete_file(file_id: str):
    """Delete an uploaded file"""
    
    # Find and delete file
    for file_path in UPLOAD_DIR.glob(f"{file_id}_*"):
        if file_path.is_file():
            file_path.unlink()
            return {"message": f"File {file_id} deleted successfully"}
    
    raise HTTPException(status_code=404, detail="File not found")

@router.get("/files")
async def list_files():
    """List all uploaded files"""
    
    files = []
    for file_path in UPLOAD_DIR.glob("*"):
        if file_path.is_file() and "_" in file_path.name:
            file_id = file_path.name.split("_")[0]
            stat = file_path.stat()
            files.append({
                "file_id": file_id,
                "filename": file_path.name.split("_", 1)[1],
                "size": stat.st_size,
                "upload_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "file_type": get_file_type(file_path.name)
            })
    
    return {"files": files, "total_count": len(files)}
