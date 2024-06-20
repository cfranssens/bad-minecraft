# Bad Minecraft
A badly made clone of the popular sandbox game Minecraft.

### Introduction
This is an attempt to recreate Minecraft in a way that is optimised. I made this project with the goal to learn more about optimisation techniques, and game development in general.

### Greedy meshing
![image](https://github.com/cfranssens/bad-minecraft/assets/101316630/66b32b70-27b5-41ba-905e-1568ad36362b)


One of the first things implemented, was greedy meshing in a compute shader to be run in parallel.
A workgroup is optimised to handle the meshing of a single 32x32x32 chunk. 

https://github.com/cfranssens/bad-minecraft/blob/master/src/shaders/generate_chunk.glsl

| Average time per dispatch per chunk | Time per dispatch in relation to dispatch size |
------------- | -------------
| ![image](https://github.com/cfranssens/bad-minecraft/assets/101316630/190a76dd-ac69-4e7a-9955-575d22f958c7)    |    ![image](https://github.com/cfranssens/bad-minecraft/assets/101316630/b66423ce-bd0a-462e-bc5e-ca97a8ff1fb8) |



Inspired by TanTan and Davis Morley

https://github.com/TanTanDev/binary_greedy_mesher_demo

https://www.youtube.com/watch?v=4xs66m1Of4A
