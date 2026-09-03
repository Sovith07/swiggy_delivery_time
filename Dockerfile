# set the base image
FROM python:3.12-slim

# install lightgbm dependency
RUN apt-get update && apt-get install -y libgomp1

# set up the working directory
WORKDIR /app



# copy the app contents
COPY flask_app/ /app/
COPY ./models/preprocessor.joblib ./models/preprocessor.joblib
COPY ./run_information.json ./
# install the packages
RUN pip install -r dev_requirements.txt



# expose the port
EXPOSE 8000

# Run the file using command
CMD [ "python","./app.py" ]