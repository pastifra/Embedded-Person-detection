# EMBEDDED PERSON DETECTION #
 
Questo lavoro di tesi si concentra su ricercare e sviluppare le tecniche di detection di persone in ambito Computer Vision su dispositivi  Embeddeed.
Nella cartella DATASETs possono essere trovati il dataset usato per il testing delle performance e una versione ridotta del dataset usato per il training.

In Tester/Stat_evaluation.ipynb possono essere analizzate le performance di alcuni algoritmi "off-the-shelf" sul testing dataset. Viene evidenziato che
una rete neurale funziona molto meglio rispetto alle precedenti soluzioni ma è molto più lento (13 secondi a frame su Pi3, 4.5 su Pi 4).

Analizzati questi limiti vengono proposte due soluzioni:

* Yolo Custom : Questa soluzione si basa sull'idea di minimizzare i layer convoluzionali e il numero di filtri della rete originale di Yolo. In particolar modo la matrice di uscita di Yolo viene ridotta da 7x7x30 a 7x7x5, ogni cella sarà quindi responsabile di una sola detection per una sola classe (Persona). Due reti sono state sviluppate; yolo_custom e yolo_custom_DW (Yolo_custom_DW utilizza layer convoluzionali meno computazionlmante complessi ma ha performance peggiori). Il codice è così organizzato nella cartella Yolo_Custom :
  * Yolo_custom_TRAIN.ipynb per effettuare il training della rete 
  * Yolo_custom_DEPLOY_Pi4.py per osservare le prestazioni della rete in real time sul raspberry Pi 4 (I due diversi modelli da caricare sono in formato .pb e si trovano nella sotto-cartella Model)
* Yolo v6 nano : Questa soluzione si basa su una rete stato dell'arte che utilizza pesi già trainati dagli autori, è stato fatto un training ulteriore per adattare la rete alla classe persona. Il codice si trova nella cartella Yolo_v6_Nano ed è così organizzato :
  * Yolo_v6n_TRAIN.ipynb per effettuare il training della rete su un dataset customizzato
  * Yolo_v6n_DEPLOY.py per osservare le prestazioni della rete sul raspberry Pi 4 (Il modello è in formato .onnx nella cartella Model e serve caricare anche il  file coco.names)

Nella cartella Tester può poi essere osservato il codice utilizzato per testare le performance delle soluzioni proposte.

<img src="/Comparison.jpg">

| Algoritmo | Yolo v3 | Yolo Custom | Yolo Custom DW | Yolo v6 nano |
| :-------: | :-----: | :---------: | :------------: | :----------: |
| **FPS**   | 0.2     | 3.2         | 4.8            | 2.5          |

Come può essere notato in figura, e osservato in tabella la soluzione Yolo v3 "off-the-shelf", avendo di gran lunga il numero di filtri e layer maggiori è quella con performance di detection migliori ma non è possibile farla funzionare in real time sul Pi 4.

La soluzione Custom ha performance peggiori, la curva scende più in fretta sintomo di scarsa precisione nelle detection proposte. (Ragioni : Pochi parametri, training difficile da fare da zero e detection proposte dalla rete solo 49). Tuttavia è la migliore come velocità.

La soluzione basata su Yolo v6 nano, fine tuned sul training dataset propone un buon compromesso tra velocità e performance di detection.

Hog è un altro algoritmo off-the shelf, che non si basa su rete neurali, implementabile molto facilmente con la libreria opencv.


## TO DO : YOLO HEATMAP ##

Develop a new type of solution based on Yolo.
The idea is to downsaple the image with convolutional layers.
Then at three resolutions (7x7, 14x14, 28x28) produce a probability prediction based on the centroid.

