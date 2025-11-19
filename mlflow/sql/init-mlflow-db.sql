-- Création de la base de données mlflow
SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

-- Note: Les permissions seront héritées de l'utilisateur mlops