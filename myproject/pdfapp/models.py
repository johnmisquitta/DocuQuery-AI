# myapp/models.py

from django.db import models

class UploadedPDF(models.Model):
    file = models.FileField(upload_to='pdf_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name
