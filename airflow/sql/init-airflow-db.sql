-- Création de la base de données mlflow
SELECT 'CREATE DATABASE airflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'airflow')\gexec

-- Note: Les permissions seront héritées de l'utilisateur mlops