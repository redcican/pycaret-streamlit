# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: eidodata-cloudbuild
  name: eidodataweb
  namespace: default
spec:
  replicas: 1
  selector:
    matchLabels:
      app: eidodata-cloudbuild
      tier: web
  strategy:
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
    type: RollingUpdate
  template:
    metadata:
      labels:
        app: eidodata-cloudbuild
        tier: web
    spec:
      containers:
        - image: gcr.io/GOOGLE_CLOUD_PROJECT/eidodata-cloudbuild:COMMIT_SHA
          imagePullPolicy: IfNotPresent
          name: eidodata-cloudbuild
          ports:
            - containerPort: 80
              protocol: TCP

---
apiVersion: v1
kind: Service
metadata:
  finalizers:
    - service.kubernetes.io/load-balancer-cleanup
  labels:
    app: eidodata-cloudbuild
  name: eidodataweb-service
  namespace: default
spec:
  clusterIP: 10.99.245.130
  externalTrafficPolicy: Cluster
  ports:
    - nodePort: 31050
      port: 80
      protocol: TCP
      targetPort: 8501
  selector:
    app: eidodata-cloudbuild
    tier: web
  sessionAffinity: None
  type: LoadBalancer