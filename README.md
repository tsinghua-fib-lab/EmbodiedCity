# Embodied City
* 1 [Introduction 🌟](#Introduction)
* 2 [Simulator 🌆](#Simulator)
* 3 [Usage 🔑](#Usage)
* 4 [Embodied Tasks 📋 ](#Tasks)


##  1 <a name='Introduction'></a> Introduction 🌟

[Embodied City](https://embodied-city.fiblab.net) is a platform that supports multiple users completing embodied tasks as drones and vehicles in an open city environment. 

![](Simulator.png)


##  2 <a name='Simulator'></a> Simulator 🌆

We construct an environment where the agents🤖 can perceive, reason, and take actions. The basic environment of the simulator includes a large business district in Beijing, one of the biggest city in China, in which we build 3D model for buildings, streets, and other elements, hosted by [Unreal Engine](https://www.unrealengine.com/).
#### 2.1 Buildings 
We first manually use [Blender4](https://www.blender.org/) to create the 3D asserts of the buildings, for which we use the streetview services of [Baidu Map](https://map.baidu.com/) and [Amap](https://amap.com/). The city level detail includes a variety of building types such as office towers🏢, shopping malls🏬, residential complexes🏠, and public facilities🏫. These models are textured and detailed to closely resemble their real-world counterparts to enhance realism in the simulation.
#### 2.2 Streets 
The streets are modeled to include all necessary components such as lanes🛣️, intersections❌, traffic signals🚦, and road markings⬆️. We also incorporate pedestrian pathways, cycling lanes, and parking areas. Data from traffic monitoring systems and mapping services help ensure that the street layout and traffic flow patterns are accurate and realistic.

#### 2.3 Other Elements 

Other elements include street furniture🚸 (benches, streetlights, signs) , vegetation🌳 (trees, shrubs, lawns), and urban amenities🚉 (bus stops, metro-entrances, public restrooms). These are also created using Blender, based on real-world references from the street view services mentioned above. Additionally, dynamic elements like vehicles🚗 and pedestrians🚶 are simulated to move realistically within the environment, contributing to the liveliness and accuracy of the urban simulation. The simulation algorithms of vehicles and pedestrians are based on [Mirage Simulation System](https://dl.acm.org/doi/pdf/10.1145/3557915.3560950).


##  3 <a name='Usage'></a> Usage 🔑
We provide A Simple Python SDK to interact with the Embodied City API.

#### 3.1 Sign Up
Before using the SDK, you need to acquire a agent and obtain its token signing up at [Embodied City](https://embodied-city.fiblab.net). In the website, you should go to the **Console** page, choose an available drone, and click on the **Acquire** button.

❗️The token is a secret key that should not be shared with anyone.
❗️The token will expire after a certain period of time if you do not use it.

#### 3.2 Installation
```bash
pip install embodied-city-python-sdk
```

#### 3.3 Initialization
```bash
from embodiedcity import DroneClient, ImageType, CameraID

base_url = "https://embodied-city.fiblab.net"
drone_id = "xxx"
token = "xxxxxxxx"
client = DroneClient(base_url, drone_id, token)
```
#### 3.4 Basic Control

```bash
# Move the drone forward by 1 meter
client.move_back_forth(1)
```
```bash
# Take a picture of the scene
image = client.take_picture(ImageType.Scene, CameraID.FrontCenter)
```

```bash
# Take a picture of the depth
image = client.take_picture(ImageType.DepthPlanar, CameraID.FrontCenter)
```
After you finish using the drone, you should release it to make it available for others. You can do this by clicking on the **Release** button in the **Console** page. Here is the detailed [API documentation](./API.py).

##  4 <a name='Tasks'></a> Embodied Tasks 📋 

In the Embodied City, we define five key embodied tasks that reflect three essential human-like abilities for intelligent agents in an open world: perception, reasoning, and decision-making. For perception, we focus on the task of embodied first-view scene understanding; for reasoning, we address embodied question answering and dialogue; and for decision-making, we include embodied action (visual-language navigation) and embodied task planning. We provide a set of [human-refined image-text datasets](./Datasets) for training and evaluating these embodied tasks.

![Embodied Tasks](./Embodied_Tasks.png)

<!-- #### 4.1 Embodied First-view Scene Understanding

The first-view scene understanding requires the agent able to well observe the environment, and give the accurate description, which could considered as a basic ability for the further tasks. We observe from different perspectives at the same location, generating a set of RGB  images, i.e., the input of scene understanding. The output is the textual description for the given scene images.

#### 4.2 Q&A

With the first-view observation, the embodied agent could be further fed with a query posed in natural language about the environment. The ***input*** includes both the first-view RGB images and a query about the environment. The ***output*** should be the direct textual responses to the question. Here we provide three questions:

1. How many traffic lights can be observed around in total?
2. Is there a building on the left side? What color is it?
3. Are you facing the road, the building, or the greenery?

#### 4.3 Dialogue

Embodied dialogue involves ongoing interactions where the agent engages in a back-and-forth conversation with the user。 This requires maintaining context and understanding the flow of dialogue. Therefore, the ***input*** includes embodied observations and multi-round queries, and the ***output*** is the multi-round responses. Here we provide three dialogues:
1. May I ask if there are any prominent waypoints around? **->** Where are they located respectively?
2. May I ask what color the building on the left is? **->** Where is it located relative to the road ahead?
3. How many trees are there in the rear view? **->** What colors are they respectively?

#### 4.4 Embodied Action (VLN)
Embodied Action, often referred to as Vision-and-Language Navigation (VLN), is a research area in artificial intelligence that focuses on enabling an agent to navigate an environment based on natural language instructions. The input combines visual perception and natural language instructions to guide the agent through complex environments. The output is the action sequences following the language instructions.

#### 4.5 Task Planning
The decision-making in the real world does not have explicit instructions; otherwise, there is only a task goal. It is significant for the embodied agents to be able to compose the complex and long-term task goals into several sub-tasks, which we refer to as embodied task planning. The ***input*** is the first-view observations and a given natural language described task goal, and the ***output*** should be a series of sub-tasks that the agent plans to execute. Here we provide three tasks and 

1. I want to have a cup of coffee at ALL-Star coffee shop, but I have not brought any money. What should I do? Please give a chain-like plan.
2. I need to get an emergency medicine from the pharmacy, but I do not know the way. What should I do? Please give a chain-like plan.
3. I lost my wallet nearby, and now I need to find it. What should I do? Please give a chain-like plan. -->
