# traffic-music
music gen based on traffic


uses yolov11
tracks speed of cars individually with diff ids. 
saves into csv with their locations and frame by frame


simply run first:,
1. cd final
run the venv
2. pip install -r requirements.txt
3. there might be some cuda download errors
    a. python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    or
    b. python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

4. python -m src.pipeline from final directory
5. it will give out the mse and weights in the weights folder (right now trains only on 1 video so MSEs are low as expected)

