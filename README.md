# The architecture of the proposed PIDC-NN, also called MinerNet-Pro. To read and download the article, please click here [PIDC-NN (MinerNet) Article](https://doi.org/10.36227/techrxiv.23266301.v3)

![PIDC-NN_MinerNet-Pro](https://github.com/REFATESHAQ/PIDC-NN_MinerNet-Pro/assets/48349737/1cfc3ddd-a529-40e7-8839-88a4447eca58)

In this work, the PIDC-NN (MinerNet) has been developed into PIDC-NN (MinerNet-Pro). Note that  the concept of PIDC-NN (MinerNet)  and PIDC-NN (MinerNet-Pro) is the same and the followings are the mean steps:
1) the number of PID units in the first layer has been increased to 36 PID units in order to increase the number of extracted features from the infrared image so the size of the image should be (240 * 108 * 3).
2)  The number of PID units in the second layer has been increased to 12 PID units.
3)  The number of PID units in the third layer has been increased to 4 PID units.
4)  The number of PID units in the fourth layer has been increased to 2 PID units.
5) the last layer has one PID unit, its output is the feedback that produces the error to adjust all extracted features from the image so the total number of extracted features by PIDC-NN (MinerNet-Pro) is 165 features. These features are exported to the fully-connected neural network for binary classification tasks.

### Important Notice

If you would like to operate this Network with other images in any other field but the number of images is different, you must adjust these numbers as shown in the below figure. For example, in my case, the number of infrared images of coal was 308 images and the number of infrared images of gangue was 308 images so the total was 616 images. The numbers of training, validation, and test images,  in the Data processing section of the code as displayed in the below figure,  were adjusted based on the number of images I had. If you do not understand what I mean and do not adjust these numbers, certainly the Network does not work well.


![Capture](https://github.com/REFATESHAQ/PIDC-NN_MinerNet-Pro/assets/48349737/a3a98096-276a-46f4-b669-a6e44dd2135e)

I emphasize that this algorithm (PIDC) that I created through my own effort, can provide optimal control to any system (not only ANN) whether linear or nonlinear with multiple inputs. Furthermore, this algorithm (PIDC) can control multiple complicated random inputs and make the system linear even with inputs, their amounts, and values are huge numbers (goes to infinity).     

The code is licensed under GNU Affero General Public License Version 3 (GNU AGPLv3); for more information, see https://www.gnu.org/licenses/agpl-3.0.en.html. The data (Coal and Gangue Infrared Images in BMP file format (Data.zip)) are licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) License. For more information, see https://creativecommons.org/licenses/by/4.0/. 

This work has been supported by my livelihood and my family's aid. The code and data are connected to the article, entitled [Deep Learning Algorithm for Computer Vision with a New Technique and Concept: PIDC-NN for Binary Classification Tasks in a Coal Preparation Plant (MinerNet)](https://doi.org/10.36227/techrxiv.23266301.v3)



