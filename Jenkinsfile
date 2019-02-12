#!groovy

pipeline {
    agent any

    stages {
        stage('Test') {
            steps {
            	sh '''
                    virtualenv testenv -p /usr/bin/python3
					sh testenv/bin/activate
					pip3 install -U pytest
					make install
					make jenkins-test
                '''
            }
        }
    }
    post {
        always {
            cleanWs()
        }
    }
}
