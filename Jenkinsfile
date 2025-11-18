pipeline {
    agent any

    environment {
        IMAGE_NAME = "dmmprice/home_backend"
        CONTAINER_NAME = "home_backend"
        PORT = "6001"
    }

    stages {

        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/DMMPrice/Home-Backend.git'
            }
        }

        stage('Load ENV File') {
            steps {
                withCredentials([file(credentialsId: 'home-backend-env', variable: 'ENV_FILE')]) {
                    sh 'cp $ENV_FILE .env'
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                sh """
                docker build -t ${IMAGE_NAME}:latest .
                """
            }
        }

        stage('Docker Login') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-dmmprice', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    sh """
                    echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                    """
                }
            }
        }

        stage('Push Image to DockerHub') {
            steps {
                sh """
                docker push ${IMAGE_NAME}:latest
                """
            }
        }

        stage('Deploy on VPS') {
            steps {
                sh """
                echo "Pulling latest image on VPS..."
                docker pull ${IMAGE_NAME}:latest

                echo "Stopping old container if exists..."
                docker stop ${CONTAINER_NAME} || true
                docker rm ${CONTAINER_NAME} || true

                echo "Starting new container..."
                docker run -d --name ${CONTAINER_NAME} \
                    --env-file .env \
                    -p ${PORT}:8000 \
                    --restart always \
                    ${IMAGE_NAME}:latest

                echo "Deployment completed on port ${PORT}"
                """
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}
