# Task Sequencing Robotics Simulator

The Task Sequencing Robotics Simulator is a Python-based simulator for (1) training robot manipulation tasks using reinforcement learning,
and (2) connecting the trained task with other trained or programmed tasks to compose a sequenced execution.

The simulator is designed to work with the Bonsai Azure Service.
The simulator is designed to work with different physics/rendering engines as well as different robot configurations.

## Samples

**Training**

To run the sample code, you will need to connect the simulator to the Bonsai Azure Service
(please copy the content of *src/samples/training/shadow_active_grasp.ink* to your Bonsai workspace).

The sample code will train a grasping brain using the PyBullet physics engine and the shadowhand_lite robotics hand.
The sample requires downloading the robot hand model from the shadow-robot/sr_common repository (melodic-devel branch).
Please use the Dockerfile for simplified setup.

```
docker image build --tag <bonsai_workspace_name>.azurecr.io/tss:1.0 -f ./docker/Dockerfile_sample .
docker run -e SIM_WORKSPACE=<bonsai_workspace_id> -e SIM_ACCESS_KEY=<access_key> --network host <bonsai_workspace_name>.azurecr.io/tss:1.0
```

To train using multiple simulator instances, please upload the built docker image to your Bonsai workspace.

```
az acr login -n <bonsai_workspace_name>
docker push <bonsai_workspace_name>.azurecr.io/tss:1.0
```

**Execution**

Please finish the above Training sample first (including building the docker image) and export a trained grasping brain.

The sample code will simulate a sequenced execution of tasks.
Please note that the current sample has some limitations where all task parameters must be defined a priori before execution.

```
docker run -d -p 5000:5000 <bonsai_workspace_name>.azurecr.io/<bonsai_workspace_id>/<trained_brain_name>:<tag>
docker run -e SIM_WORKSPACE=w -e SIM_ACCESS_KEY=a -e RUNCODE=src/samples/execution/pick_place.py --network host <bonsai_workspace_name>.azurecr.io/tss:1.0
```

A video of the execution is saved under logs/ inside the docker container.
(There are ways to run the container such as using *tail -f /dev/null* which could make accessing the logs easier.)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
