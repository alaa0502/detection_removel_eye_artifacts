Artifact is a signal, caused by an extracerebral source, observed during EEG recording and is considered unwanted information (noise) in brain studies.
 The common method of removing artifacts is done manually by researchers and is time- and effort-consuming.
 Therefore, a tool that performs this automatically is needed.
 
This web app is provided to offer a friendly, automatic, and reliable tool for (currently) non-expert use and students.
 All that is required is to upload an EEG file, choose the strictness of cleaning (default or fine-tuned), and click Start. A cleaned version of the data is then made available for download, ready for use.
 
 visit:https: //eye-artifact-cleaner.streamlit.app/
 
The tool’s parameters were carefully chosen after a training process using different methods (RNN, CNNs) and were tested with real data, approaching ~96% accuracy and ~0.96 AUC-PR, and are supported by scientific studies.
However, it is still under development to match professional use, since training was performed mainly on data selected from Fp1 and Fp2 channels (sufficient for blink artifacts). Coverage is planned to be expanded to all eye artifacts (voluntary and involuntary), and then to be extended gradually to other physiological, technical, and environmental artifacts.

To keep things simple, all information regarding both training models is included in the code.
In this project environment, access is provided to the web tool, the web code, the RNN model (the chosen one), and the CNNs model.
