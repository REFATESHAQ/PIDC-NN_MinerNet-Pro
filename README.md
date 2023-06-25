# The architecture of the proposed PIDC-NN, also called MinerNet-Pro. To read and download the article, please click here [PIDC-NN_MinerNet Article](https://doi.org/10.36227/techrxiv.23266301.v2)
![PIDC-NN_MinerNet-Pro](https://github.com/REFATESHAQ/PIDC-NN_MinerNet-Pro/assets/48349737/5a7f0902-913c-44ea-91d8-fe09e072e0dc)

In this work, the PIDC-NN (MinerNet) has been developed into PIDC-NN (MinerNet-Pro). Note that  the concept of PIDC-NN (MinerNet)  and PIDC-NN (MinerNet-Pro) is same and the followings are the mean steps:
1) the number of PID units in the first layer has been increased to 36 PID units in order to increase the number of extracted features from the infrared image so the size of the image should be (240*108*3).
2)  The number of PID units in the second layer has been increased to 12 PID units.
3)  The number of PID units in the third layer has been increased to 4 PID units.
4)  The number of PID units in the fourth layer has been increased to 2 PID units.
5) the last layer has one PID unit, its output is the feedback that produces the error to adjust all extracted features from the image so the total number of extracted features by PIDC-NN (MinerNet-Pro) is 165 features. These features are exported to the fully-connected neural network for binary classification tasks.
![Capture](https://github.com/REFATESHAQ/PIDC-NN_MinerNet-Pro/assets/48349737/ccfb5032-ba1d-4e79-be9c-2c1812aef1f7)
