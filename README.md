# Internship_code
Code developed during the internship at armasuisse S&amp;T at Thun

The devices where organized in a local network with the following IPs:
- Rasp 4: *192.168.0.7-16*
- Rasp 3: *192.168.0.17-26* 
- Rasp 1: *192.168.0.27-36*
- Rasp Zero *192.168.0.37-47*

*command_RPiXXXX.sh* scripts help to send commands to all the devices of the same model/same GPU

*launch_data_collection.sh* and *gather_data.sh* are used to start the data collection campaigns and retrieve the data afterwards.
  *launch_data_collection_stress.sh* launches the data collection campaign together with *stress-ng* command exhaustively using device resources.

*quick_ssh.sh* just wraps the ssh connection generation by receiving the last IP number, avoiding to write the pass in each connection.

*TREASURE* folder is the one to be placed in the RPi devices, it requires to have py-videocore (RPi Zero/1/2/3) or py-videocore6 (RPi 4) installed.

*/boot/ config.txt* and *cmdline.txt* should be placed in /boot partition of the SD to enable turbo_mode and isolate CPU 3.

*datasets* folder contains the preliminary datastets generated during the data collection.

*experiments* folder contains the experiments carried out with the gathered data.

*wpa_supplicant.conf* should be added in /boot/ of Rasp Zero to enable Wifi connectivity.
