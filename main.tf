provider "google" {
  project     = "circle-425516"
  region      = "us-west2"
}

resource "google_storage_bucket" "my_bucket" {
  name                     = "circle-terraform-bucket"
  location                 = "US"
  force_destroy            = true
  public_access_prevention = "enforced"
  project                  = "circle-425516"
}
