from config.supabase_config import supabase
import uuid

def upload_to_supabase(file, format="png"):
    try:
        unique_id = uuid.uuid4().hex
        filename = f"result_{unique_id}.{format}"
        upload_res = supabase.storage.from_('Uploads').upload(filename, file, file_options={"content-type": f"image/{format}", "upsert": "false"})

        public_url = supabase.storage.from_('Uploads').get_public_url(upload_res.path)
        
        return {"status": "SUCCESS", "result_url": public_url, "result_path": upload_res.path }
    except Exception as uploadErr:
        print(f"Couldn't upload the result image to storage: {str(uploadErr)}")
        return {"status": "FAILED", "error_code": "UPLOAD_ERROR"}




