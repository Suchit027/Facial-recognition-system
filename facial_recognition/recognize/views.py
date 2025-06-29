# recognize/views.py
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from .models import Person
from .face_utils import extract_embedding, match_embedding

from django.shortcuts import render
from django.http import JsonResponse, HttpResponse

def register_person(request):
    if request.method == "POST":
        name = request.POST.get('name')
        image = request.FILES.get('image')
        
        if not name or not image:
            return JsonResponse({'error': 'Name and image are required.'}, status=400)

        # Save person and image
        from .models import Person
        person = Person(name=name, image=image)
        embedding  = extract_embedding(image)
        person.set_embedding(embedding)
        person.save()

        return JsonResponse({'message': 'Person registered successfully!'})
    
    return render(request, 'recognize/register.html')




def identify_person(request):
    if request.method == "POST":
        image = request.FILES["image"]
        emb = extract_embedding(image)
        persons = Person.objects.all()
        match, score = match_embedding(emb, persons)
        if match:
            return JsonResponse({"match": match.name, "score": score})
        return JsonResponse({"match": None, "score": score})
    return render(request, 'recognize/identify.html')