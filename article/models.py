from django.db import models

# Create your models here.

class users(models.Model):
	name = models.TextField()
	email = models.TextField()
	message = models.TextField()