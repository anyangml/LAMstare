# LAMstare
<p align="center">
<img width="125" alt="image" src="https://github.com/user-attachments/assets/e4949ee3-1ec6-4eab-b817-b1596035bcc2">
</p>

LAMstare is an MLOps tool developed for monitoring the training process of Large Atomic Models (LAM) within the deepmd-kit framework. It serves two primary functions:
- Monitoring the training process (for both single-task and multi-task scenarios) with a learning curve and on-the-fly in-distribution (ID) tests.
- Conducting thorough evaluations of the model on out-of-distribution (OOD) tests.



## How to use

The MLOps pipeline is designed to be as straightforward as possible. To monitor the training process, you can set up cron jobs. The less frequent OOD tests should be triggered manually.

- To monitor the training process, set a cron job for `LAMstare/lamstare/experiments/cron_job.sh`.
- Update the experiments you wish to monitor in `LAMstare/lamstare/utils/dlc_submit.py`.