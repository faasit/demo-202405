# Setup and Usage
1. Set up k8s environment on both master and worker machine.
2. Make sure the images is pulled on the worker machines, or the latency test will include time for image pulling and output unexpected log info.
```bash
docker pull redis
docker pull enavenue/hotelpipe-img
```
3. go to base dir and run following commands, which will start testing this application under both traditional serverless framework and serverless pilot.
```bash
bash exp_scripts/moviepipe.sh
```
# Application Overview
This web application contains seven stages, simulating a naive web server receiving request from user interacting with a movie review website.
1. Movieinfo: Generate random movie information and user information.
2. Request: Generate user request of four types: user-login, update-review, recommend-movie and search-movie.  
3. Userlogin: Testify whether user id matches its password.
4. Reviewupdate: Simulate user writing movie reviews and posting to the website.
5. Recommend: Recommend movie with high rates.
6. Search: Search sepcific movie with name and feed back its information.
7. Collect: Collect all the results of these requests.
This seven stages keeps sending and processing request until reaching enough time, making up entire workflow.