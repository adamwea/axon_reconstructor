### Setting HDF5_PLUGIN_PATH Permanently

**Step 1: Download and Extract the Plugin**
1. **Download the Plugin directly from here:**
   [Download Plugin](https://share.mxwbio.com/d/4742248b2e674a85be97/) Ideally save it in your $HOME directory.
   
2. **Extract the Plugin:**
   ```bash
   unzip Linux.zip -d ~/maxwell_hdf5_plugin
   ```
   This will create a directory named `maxwell_hdf5_plugin` in your home directory. Adjust the path if you extract it elsewhere.

**Step 2: Set HDF5_PLUGIN_PATH Permanently**
- For **Bash Users** (.bashrc):
   1. Add the environment variable to `.bashrc`:
   ```bash
   echo "export HDF5_PLUGIN_PATH=\$HOME/maxwell_hdf5_plugin/Linux" >> ~/.bashrc
   ```
   2. Apply the changes:
   ```bash
   source ~/.bashrc
   ```

Now, the `HDF5_PLUGIN_PATH` is set permanently, and you can use the plugin in your HDF5 applications. Enjoy your coding!