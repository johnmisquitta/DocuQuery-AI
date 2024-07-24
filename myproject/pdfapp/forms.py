# myapp/forms.py

from django import forms

class PDFUploadForm(forms.Form):
    file = forms.FileField()
    query = forms.CharField(widget=forms.TextInput(attrs={'placeholder': 'Enter your query here...'}))
