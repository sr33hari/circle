provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

data "external" "get_artifact_registry_repo" {
  program = ["bash", "gcp.sh"]
  query = {
    project_id    = var.project_id
    region        = var.region
    repository_id = var.repository_id
  }
}

output "artifact_registry_repo" {
  value = data.external.get_artifact_registry_repo.result.name
}

resource "google_artifact_registry_repository" "my_repo" {
  provider      = google-beta
  project       = var.project_id
  location      = var.region
  repository_id = var.repository_id
  format        = "DOCKER"
  labels = {
    env = "main"
  }

  count = data.external.get_artifact_registry_repo.result.name == "" ? 1 : 0
}

resource "null_resource" "build_and_push_image" {
  provisioner "local-exec" {
    command = <<EOT
      gcloud auth configure-docker ${var.region}-docker.pkg.dev
      docker build -t ${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_id}/circle-app:${var.image_tag} ./
      docker push ${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_id}/circle-app:${var.image_tag}
    EOT
  }
}

resource "google_compute_instance" "circle_instance" {
  name         = "circle-instance"
  machine_type = "e2-custom-8-16384"
  zone         = var.zone

  boot_disk {
    auto_delete = true
    device_name = "circle-instance"

    initialize_params {
      image = "projects/cos-cloud/global/images/cos-109-17800-218-69"
      size  = 50
      type  = "pd-balanced"
    }

    mode = "READ_WRITE"
  }

  can_ip_forward      = false
  deletion_protection = false
  enable_display      = false

  labels = {
    container-vm = "cos-109-17800-218-69"
    goog-ec-src  = "vm_add-tf"
  }

  metadata = {
    gce-container-declaration = <<-EOF
      spec:
        containers:
        - name: circle-instance
          image: ${var.region}-docker.pkg.dev/${var.project_id}/${var.repository_id}/circle-app:${var.image_tag}
          env:
          - name: MONGO_URI
            value: ${var.MONGO_URI}
          - name: GOOGLE_TYPE
            value: ${var.GOOGLE_TYPE}
          - name: GOOGLE_PROJECT_ID
            value: ${var.GOOGLE_PROJECT_ID}
          - name: GOOGLE_PRIVATE_KEY_ID
            value: ${var.GOOGLE_PRIVATE_KEY_ID}
          - name: GOOGLE_PRIVATE_KEY
            value: ${var.GOOGLE_PRIVATE_KEY}
          - name: GOOGLE_CLIENT_EMAIL
            value: ${var.GOOGLE_CLIENT_EMAIL}
          - name: GOOGLE_CLIENT_ID
            value: ${var.GOOGLE_CLIENT_ID}
          - name: GOOGLE_AUTH_URI
            value: ${var.GOOGLE_AUTH_URI}
          - name: GOOGLE_TOKEN_URI
            value: ${var.GOOGLE_TOKEN_URI}
          - name: GOOGLE_AUTH_PROVIDER_X509_CERT_URL
            value: ${var.GOOGLE_AUTH_PROVIDER_X509_CERT_URL}
          - name: GOOGLE_CLIENT_X509_CERT_URL
            value: ${var.GOOGLE_CLIENT_X509_CERT_URL}
          - name: GOOGLE_UNIVERSE_DOMAIN
            value: ${var.GOOGLE_UNIVERSE_DOMAIN}
          stdin: false
          tty: false
        restartPolicy: OnFailure
    EOF
    google-logging-enabled    = "true"
  }

  network_interface {
    network    = "default"
    subnetwork = "projects/${var.project_id}/regions/${var.region}/subnetworks/default"

    access_config {
      network_tier = "PREMIUM"
    }
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "MIGRATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  service_account {
    email  = var.service_account_email
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }

  shielded_instance_config {
    enable_integrity_monitoring = true
    enable_secure_boot          = false
    enable_vtpm                 = true
  }

  tags = ["http-server", "https-server"]

  depends_on = [null_resource.build_and_push_image]
}

