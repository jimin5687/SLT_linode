from django.shortcuts import render, redirect
from django.views.generic import TemplateView
from django.urls import reverse_lazy

from mysite.camera import VideoCamera, gen
from django.http import StreamingHttpResponse


class Home(TemplateView):
    template_name = 'base.html'



# for video input and detection
# the whole thing, video
# is returned as a streaming http response, or bytes
def video_stream(request):
    # Define the streaming content generator function
    def stream_content():
        # You should implement your generator function (gen) here
        cam = VideoCamera()
        yield from gen(cam)

    # Return a StreamingHttpResponse with the streaming content
    return StreamingHttpResponse(stream_content(), content_type='multipart/x-mixed-replace;boundary=frame')

# def render_camera_stream(request):
    # Render the HTML template
    return render(request, 'camera_stream.html')

# def video_input(request):
#     return render(request, 'camera.html')