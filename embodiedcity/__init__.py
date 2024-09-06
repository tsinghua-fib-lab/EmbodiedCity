"""
A Simple Python SDK to interact with the Embodied City API.
Users can easily achieve perception and control of drone agents through the following functions.
When the command is issued via the API, changes in the agent's first-person view will be observed in the Console.

## Installation

```bash
pip install embodiedcity
```

## Usage

#### Acquire ID and token

Before you can use the SDK, you need to acquire a drone and obtain its token.
You can get one by signing up at [Embodied City](https://embodied-city.fiblab.net/).

In the website, you should go to the "Console" page, choose an available drone, and click on the "Acquire" button.
After that, you will see a token with the drone ID.

> ATTENTION: The token is a secret key that should not be shared with anyone.

> ATTENTION: The token will expire after a certain period of time if you do not use it. (the time constrain will be notified in the website)

#### Initialize the client

```python
from embodiedcity import DroneClient, ImageType, CameraID

base_url = "https://embodied-city.fiblab.net"
drone_id = "xxx"
token = "xxxxxxxx"
client = DroneClient(base_url, drone_id, token)
```

#### Move the drone

```python
# Move the drone forward by 10 meter (Short movement distance may result in action failure)
client.move_back_forth(10)
```

#### Obtain the RGB image of the front camera
    
```python
# Get a RGB image from the front-center camera
image = client.get_image(ImageType.Scene, CameraID.FrontCenter)
```

#### Get the depth image
    
```python
# Get an image of the depth from the front-center camera
image = client.get_image(ImageType.DepthPlanar, CameraID.FrontCenter)
```

## Release the drone

After you finish using the drone, you should release it to make it available for others.

You can do this by clicking on the "Release" button in the "Console" page.


## FAQ

#### After invoking the control action, the drone did not move.

It is possible that the drone collided with a building.
Try issuing a command to move the drone in a direction without obstacles.
Alternatively, use the function DroneClient.move_to_position to force it to a specified location.

#### What should I do if I need the drone to perform more complex operations?

Please download and install the full embodiedcity simulator.

"""

from .client import DroneClient, ImageType, CameraID

__all__ = ["DroneClient", "ImageType", "CameraID"]
