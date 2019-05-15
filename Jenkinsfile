#!groovy

void setBuildStatus(String message, String state) {
  step([
      $class: "GitHubCommitStatusSetter",
      reposSource: [$class: "ManuallyEnteredRepositorySource", url: "https://github.com/brg-jenkins/pypsl"],
      contextSource: [$class: "ManuallyEnteredCommitContextSource", context: "ci/jenkins/build-status"],
      errorHandlers: [[$class: "ChangingBuildStatusErrorHandler", result: "UNSTABLE"]],
      statusResultSource: [ $class: "ConditionalStatusResultSource", results: [[$class: "AnyBuildResult", message: message, state: state]] ]
  ]);
}

pipeline {
    agent any

    stages {
        stage('Preparing environment') {
            steps {
                setBuildStatus("Build in progress", "PENDING");
                sh 'docker build --tag=testimage .';
            }
        }
        stage('Test') {
            steps {
               sh "docker run -v ${WORKSPACE}:/tmp testimage sh -c 'make install; pytest tests/'";
            }
        }
        stage('Coverage') {
            when {
                branch 'master'
            }
            steps {
               sh "docker run -v ${WORKSPACE}:/tmp testimage sh -c 'make install; coverage run -m pytest tests/; coverage-badge -o coverage.svg'"
               s3Upload(file:'coverage.svg', bucket:'pypsl-public', path:"cov_${env.BRANCH_NAME}.svg")
            }
        }
    }
    post {
        success {
            setBuildStatus("Build succeeded", "SUCCESS");
        }
        failure {
            setBuildStatus("Build failed", "FAILURE");
        }
        always {
            cleanWs()
        }
    }
}
