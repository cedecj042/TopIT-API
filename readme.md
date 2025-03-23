# TopIT API Setup

## Prerequisites

Ensure you have the following installed before proceeding:

1. **WSL (Windows Subsystem for Linux)** (For Windows users)
2. **Ubuntu** (For Windows users via WSL)
3. **Docker Desktop** (For containerization)

## Setup Instructions

### Step 1: Configure Docker with WSL (For Windows Users)

1. Open **Docker Desktop**.
2. Navigate to **Settings** → **Resources** → **WSL Integration**.
3. Enable the option:

   ```
   Enable integration with my default WSL distro
   ```

4. Click **Apply & Restart**.

### Step 2: Clone the Repository

```sh
# Inside your Ubuntu terminal:
git clone <your-repository-url>
cd <your-repository-name>
```

### Step 3: Set Up Environment Variables

Create a `.env` file in the project root and configure your environment variables as required:

```sh
cp .env.example .env
nano .env  # Edit as needed
```

### Step 4: Build and Start the Containers

```sh
docker-compose up -d --build
```

### Step 5: Verify Services

Ensure all containers are running:

```sh
docker ps
```

Test the API by making a request:

```sh
curl -X GET http://localhost:8001/docs
```

Or open the Swagger UI in your browser:

```
http://localhost:8001/docs
```

### Step 6

## Additional Commands

- **Rebuild the containers** (after making changes to `Dockerfile` or dependencies):

    ```sh
    docker-compose up -d --build
    ```

- **Stopping the Containers**
To stop the containers, run:

    ```sh
    docker-compose down
    ```

## Troubleshooting

1. **ChromaDB Connection Issues:** Ensure `chromadb` is in the same network as `app` or `laravel`:

   ```sh
   docker network inspect topit
   ```

2. **Recreate the Network if Needed:**

   ```sh
   docker network create topit
   ```

3. **Manually Connect Containers to the Network:**

   ```sh
   docker network connect topit chromadb
   ```

For further debugging, check container logs:

```sh
docker logs <container_name>
```