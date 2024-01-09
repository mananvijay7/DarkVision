from django import forms
from .models import Images

#DataFlair
class ImagesIn(forms.ModelForm):

	class Meta:
		model = Images
		fields = '__all__'
		