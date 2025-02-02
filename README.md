# OCR
An optical character recognition program using pytorch
Run the following beforehand to ensure all librries are installed:
```
pip install torch torchvision matplotlib keyboard textblob opencv-python pillow
```

Place a file 'input.png' in the same directory as the three python files and the save file and run to extract text. 
> [!NOTE]
> This works with printed and handwritten text that isn't connected. 
> For the original purpose of this program, the switch case statement (main.py lines 12-17) removes all numbers.

> [!IMPORTANT]
> Support for any other character (outside numbers, lowercase and uppercase letters) is not included.
> The program detected 8812 characters correctly out of 10000 characters from the EMNIST dataset.
> Most issues come from ambiguities (o, O and 0 etc.) or from bad handwriting.
> An external spell checker (TextBlob) is used to remove minor errors, but there are still many flaws in outputs.
