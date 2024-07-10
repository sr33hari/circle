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

variable "GOOGLE_TYPE" {
  description = "The type of the service account key"
  type        = string
}

variable "GOOGLE_PROJECT_ID" {
  description = "The ID of the Google Cloud project"
  type        = string
}

variable "GOOGLE_PRIVATE_KEY_ID" {
  description = "The ID of the private key"
  type        = string
}

variable "GOOGLE_PRIVATE_KEY" {
  description = "The private key"
  type        = string
}

variable "GOOGLE_CLIENT_EMAIL" {
  description = "The email of the service account"
  type        = string
}

variable "GOOGLE_CLIENT_ID" {
  description = "The ID of the service account"
  type        = string
}

variable "GOOGLE_AUTH_URI" {
  description = "The URI of the authentication server"
  type        = string
}

variable "GOOGLE_TOKEN_URI" {
  description = "The URI of the token server"
  type        = string
}

variable "GOOGLE_AUTH_PROVIDER_X509_CERT_URL" {
  description = "The URL of the provider's x509 certificate"
  type        = string
}

variable "GOOGLE_CLIENT_X509_CERT_URL" {
  description = "The URL of the client's x509 certificate"
  type        = string
}

variable "GOOGLE_UNIVERSE_DOMAIN" {
  description = "The domain of the universe"
  type        = string
}
