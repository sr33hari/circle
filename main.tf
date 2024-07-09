provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

data "google_artifact_registry_repository" "existing_repo" {
  provider      = google-beta
  project       = var.project_id
  location      = var.region
  repository_id = var.repository_id
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

  count = length(data.google_artifact_registry_repository.existing_repo.repository_id) == 0 ? 1 : 0
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
            value: ${var.mongo_uri}
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

