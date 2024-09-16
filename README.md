# ASL Translator v2
 
Sign language is crucial for communication with and between individuals who are hearing impaired. Being able to translate this in real-time allows individuals to be able to communicate without the sign language barrier.

The program uses Google’s MediaPipe to identify hand landmarks in the OpenCV camera feed. Using these landmarks, it crops the frame into a square that contains the entire hand with extra padding (currently multiplying the range of landmarks by a factor of 1.25 in this demo). This analysis frame is then scaled down to 28x28 and grayscaled to feed into a custom-trained neural network model.

The neural network is a CNN that was trained on the Sign Language MNIST Dataset with almost 35,000 28x28 grayscale images of signed letters. This model is an upgrade from my last version since it was trained for longer and contains a more complex model structure. The improved model has 4 blocks using ReLU activations.

The LiveDeterminer determines letters in real-time, but can be paired with a custom stringing algorithm to identify gaps in letters to automatically string letters into words. Taking the derivative of confidences allows the code to identify when a letter changes into another and logs it as a separation. Summing up the confidence matrices in between separations allows to find the top letter guesses.





https://github.com/user-attachments/assets/8d18897f-5b9b-4f0e-8965-b8d6e5a66e07


