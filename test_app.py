import os
import io
import pytest
from app import app, model_predict

@pytest.fixture
def client():
    app.config['TESTING'] = True
    client = app.test_client()
    yield client

def test_home(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.data == b"Server is running!"

def test_image_classification_route(client):
    response = client.get('/Image_Classification')
    assert response.status_code == 200

def test_predict_post(client, mocker):
    data = {
        'file': (io.BytesIO(b'test image data'), '.\test_image\spot-_0_5088.jpg')
    }
    mocker.patch('app.model_predict', return_value='Mocked_Label')
    response = client.post('/predict', content_type='multipart/form-data', data=data)
    assert response.status_code == 200
    assert response.data == b"Mocked_Label"

