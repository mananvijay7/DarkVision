from django.db import models

# Validators
def validate_file_extension(value):
    import os
    from django.core.exceptions import ValidationError
    ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
    valid_extensions = ['.arw']
    if not ext.lower() in valid_extensions:
        raise ValidationError(u'Unsupported file extension.')

# Images model
class Images(models.Model):
	picture = models.FileField(upload_to="documents/%Y/%m/%d", 
		validators=[validate_file_extension])


