variable "project_id" {
  description = "The ID of the Google Cloud project"
  type        = string
}

variable "region" {
  description = "The region where resources will be deployed"
  type        = string
}

variable "zone" {
  description = "The zone where the Compute Engine instance will be deployed"
  type        = string
}

variable "repository_id" {
  description = "The ID of the Artifact Registry repository"
  type        = string
}

variable "service_account_email" {
  description = "Email of the service account to be used"
  type        = string
}

variable "image_tag" {
  description = "Tag for the Docker image"
  type        = string
  default     = "latest"
}

variable "mongo_uri" {
  description = "MongoDB URI for the application"
  type        = string
}