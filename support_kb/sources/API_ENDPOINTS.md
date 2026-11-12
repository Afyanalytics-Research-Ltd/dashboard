# Afya Analytics HMIS 3.0 — Backend API endpoint reference

Generated from the erp-platform-3.0 route files. Each service is a separate backend; the frontend reaches it at the address set in its `*_SERVICE_URL` variable (which ends in `/api/v1`). Every endpoint except `/health` needs a signed-in user token, and most also need the permission shown. A permission like `permission:a|b` means the user needs any one of them; superadmin always passes. If a user gets 403 Forbidden on an action, the permission listed for that endpoint is missing from their role.

## Core service API (core-service)

Covers: sign-in, users, roles, permissions, organizations, facilities, settings, dashboards. 322 endpoints.

### Core service — `apps` endpoints

Area: confirmed the role check was reachable live. signed.app only now.

- `POST /v1/apps/register` → AppRegistrationController@store
- `PATCH /v1/apps/{appId}/permissions` → AppRegistrationController@updatePermissions

### Core service — `audit-logs` endpoints

Area: Notification routes

- `GET /v1/audit-logs/statistics` (needs permission:audit.view)
- `POST /v1/audit-logs/cleanup` (needs permission:audit.view; permission:system.maintenance)
- `GET /v1/audit-logs/for-model` (needs permission:audit.view)
- `GET /v1/audit-logs/for-user/{userId}` (needs permission:audit.view)
- `GET /v1/audit-logs/{id}` (needs permission:audit.view)
- `GET /v1/audit-logs` → AuditLogController@index (needs permission:audit.view)

### Core service — `collection-methods` endpoints

Area: Sample Lookup Data (consumed by frontend via CORE_SERVICE_URL/v1/*)

- `GET /v1/collection-methods` → SampleCollectionMethodController@index

### Core service — `confirm-password` endpoints

Area: "facility.access",

- `POST /v1/confirm-password` → AuthController@confirmPassword

### Core service — `dashboard` endpoints

Area: Dashboard routes

- `GET /v1/dashboard` → DashboardController@index
- `GET /v1/dashboard/users`
- `GET /v1/dashboard/modules`
- `GET /v1/dashboard/system-health`
- `GET /v1/dashboard/recent-activities`
- `GET /v1/dashboard/quick-stats`

### Core service — `employee-categories` endpoints

Area: Employee Management routes

- `GET /v1/employee-categories/{id}/employees`
- `GET /v1/employee-categories` → EmployeeCategoryController@index
- `POST /v1/employee-categories` → EmployeeCategoryController@store
- `GET /v1/employee-categories/{id}` → EmployeeCategoryController@show
- `PUT/PATCH /v1/employee-categories/{id}` → EmployeeCategoryController@update
- `DELETE /v1/employee-categories/{id}` → EmployeeCategoryController@destroy

### Core service — `employee-specialities` endpoints

Area: Employee Specialities routes

- `GET /v1/employee-specialities/{id}/employees`
- `GET /v1/employee-specialities` → EmployeeSpecialityController@index
- `POST /v1/employee-specialities` → EmployeeSpecialityController@store
- `GET /v1/employee-specialities/{id}` → EmployeeSpecialityController@show
- `PUT/PATCH /v1/employee-specialities/{id}` → EmployeeSpecialityController@update
- `DELETE /v1/employee-specialities/{id}` → EmployeeSpecialityController@destroy

### Core service — `forgot-password` endpoints

Area: AppServiceProvider::configureRateLimiting(), keyed the same way.

- `POST /v1/forgot-password` → AuthController@forgotPassword

### Core service — `forgot-password-otp` endpoints

Area: AppServiceProvider::configureRateLimiting(), keyed the same way.

- `POST /v1/forgot-password-otp` → AuthController@forgotPasswordOtp

### Core service — `gateway` endpoints

Area: do not remove one without the other.

- `POST /v1/gateway` → ModelGatewayController@handle

### Core service — `health` endpoints

Area: Health check endpoint (unversioned)

- `GET /health` → HealthController@check

### Core service — `locations` endpoints

Area: existing caller to stay backward-compatible with.; human user session on these requests. See EXTERNAL_API.md §6 Phase 2.

- `GET /v2/locations` → LocationController@index
- `GET /v2/locations/{id}` → LocationController@show
- `POST /v2/locations` → LocationController@store

### Core service — `login` endpoints

Area: AppServiceProvider::configureRateLimiting(), keyed the same way.

- `POST /v1/login` → AuthController@login

### Core service — `logout` endpoints

Area: "facility.access",

- `POST /v1/logout` → AuthController@logout

### Core service — `logout-all` endpoints

Area: "facility.access",

- `POST /v1/logout-all` → AuthController@logoutAll

### Core service — `modules` endpoints

Area: Module Management routes

- `POST /v1/modules/{id}/enable`
- `POST /v1/modules/{id}/disable`
- `POST /v1/modules/{id}/health-check`
- `POST /v1/modules/health-check-all`
- `GET /v1/modules/{id}/dependencies`
- `GET /v1/modules` → ModuleController@index
- `POST /v1/modules` → ModuleController@store
- `GET /v1/modules/{id}` → ModuleController@show
- `PUT/PATCH /v1/modules/{id}` → ModuleController@update
- `DELETE /v1/modules/{id}` → ModuleController@destroy

### Core service — `notifications` endpoints

Area: Notification routes

- `POST /v1/notifications/mark-all-read`
- `DELETE /v1/notifications/read`
- `GET /v1/notifications/unread-count`
- `GET /v1/notifications/preferences`
- `PUT /v1/notifications/preferences`
- `POST /v1/notifications/broadcast`
- `POST /v1/notifications/notify-role`
- `POST /v1/notifications/{id}/mark-read`
- `POST /v1/notifications/{id}/mark-unread`
- `GET /v1/notifications` → NotificationController@index
- `POST /v1/notifications` → NotificationController@store
- `GET /v1/notifications/{id}` → NotificationController@show
- `PUT/PATCH /v1/notifications/{id}` → NotificationController@update
- `DELETE /v1/notifications/{id}` → NotificationController@destroy

### Core service — `organizations` endpoints

Area: existing caller to stay backward-compatible with.; human user session on these requests. See EXTERNAL_API.md §6 Phase 2.

- `GET /v2/organizations` → FhirOrganizationController@index
- `GET /v2/organizations/{id}` → FhirOrganizationController@show
- `POST /v2/organizations` → FhirOrganizationController@store

### Core service — `permissions` endpoints

Area: outside that config file.

- `GET /v1/permissions/stream` → PermissionController@stream (needs permission:roles.view|users.roles.view)
- `GET /v1/permissions` → PermissionController@index (needs permission:roles.view|users.roles.view)
- `GET /v1/permissions/{id}` → PermissionController@show (needs permission:roles.view|users.roles.view)

### Core service — `practitioners` endpoints

Area: existing caller to stay backward-compatible with.; human user session on these requests. See EXTERNAL_API.md §6 Phase 2.

- `GET /v2/practitioners` → PractitionerController@index
- `GET /v2/practitioners/{id}` → PractitionerController@show
- `POST /v2/practitioners` → PractitionerController@store

### Core service — `register` endpoints

Area: request (2026-09-21) — revisit separately.

- `POST /v1/register` → AuthController@register

### Core service — `resend-2fa` endpoints

Area: AppServiceProvider::configureRateLimiting(), keyed the same way.

- `POST /v1/resend-2fa` → AuthController@resendTwoFactorCode

### Core service — `resend-verification` endpoints

Area: AppServiceProvider::configureRateLimiting(), keyed the same way.

- `POST /v1/resend-verification` → AuthController@resendVerification

### Core service — `reset-password` endpoints

Area: AppServiceProvider::configureRateLimiting(), keyed the same way.

- `POST /v1/reset-password` → AuthController@resetPassword

### Core service — `roles` endpoints

Area: Sync role permissions; populate role pickers; admins hold roles.view so they pass too.

- `GET /v1/roles` → RoleController@index (needs permission:roles.view|users.roles.view)
- `GET /v1/roles/{role}` → RoleController@show (needs permission:roles.view|users.roles.view)
- `POST /v1/roles/{role}/permissions/sync` (needs role:superadmin|admin)
- `POST /v1/roles/{role}/permissions/unassign` (needs role:superadmin|admin)
- `POST /v1/roles` → RoleController@store (needs role:superadmin|admin)
- `PUT/PATCH /v1/roles/{id}` → RoleController@update (needs role:superadmin|admin)
- `DELETE /v1/roles/{id}` → RoleController@destroy (needs role:superadmin|admin)

### Core service — `sample-types` endpoints

Area: Sample Lookup Data (consumed by frontend via CORE_SERVICE_URL/v1/*)

- `GET /v1/sample-types` → SampleTypeController@index

### Core service — `service` endpoints

Area: is how inventory-service then checks that token on every call.; routes issue — which must never be able to issue more.; that platform's plaintext secret itself. See EXTERNAL_API.md §3.2.; v1 group above; VerifySignedRequest is the only guard here.; v1/service group: a caller needs send_notifications on its RegisteredApp.

- `GET /v1/service/organizations` → ServiceOrganizationController@index
- `POST /v1/service/organizations` → ServiceOrganizationController@store
- `POST /v1/service/organizations/{id}/seed-reference-data` → ServiceOrganizationController@seedReferenceData
- `POST /v1/service/facilities` → ServiceFacilityController@store
- `GET /v1/service/facilities/{id}` → ServiceFacilityController@show
- `POST /v1/service/users` → ServiceUserController@store
- `POST /v1/service/users/{user}/assign-roles` → ServiceUserController@assignRoles
- `PATCH /v1/service/users/{user}` → ServiceUserController@update
- `PATCH /v1/service/users/{user}/password` → ServiceUserController@updatePassword
- `GET /v1/service/settings/{key}` → ServiceSettingsController@show
- `POST /v1/service/apps/verify-signature` → ServiceAppController@verifySignature
- `POST /v1/service/fieldmatch/login` → ServiceFieldMatchLoginController@login
- `POST /v1/service/fieldmatch/logout` → ServiceFieldMatchLoginController@logout
- `POST /v1/service/apps/verify-token` → ServiceFieldMatchLoginController@verifyToken
- `POST /v1/service/organizations/{organizationId}/notifications/notify-permission` → ServiceNotificationController@notifyPermission
- `GET /v1/service/organizations/{organizationId}/api-credentials` → ServiceApiCredentialController@index
- `POST /v1/service/organizations/{organizationId}/api-credentials` → ServiceApiCredentialController@store
- `POST /v1/service/organizations/{organizationId}/api-credentials/{appId}/reset-password` → ServiceApiCredentialController@resetPassword
- `PATCH /v1/service/organizations/{organizationId}/api-credentials/{appId}` → ServiceApiCredentialController@update

### Core service — `setup` endpoints

Area: "registry-search" is never read as a {facility} id.; Category Filters; Departments; Geographic Configuration; Insurance Configuration; Integration Settings

- `GET /v1/setup/settings/public`
- `POST /v1/setup/settings/bulk`
- `GET /v1/setup/settings/bulk-configs`
- `PUT /v1/setup/settings/bulk` (needs permission:settings.edit|settings.manage)
- `GET /v1/setup/settings/stream`
- `GET /v1/setup/settings` → SystemSettingController@index
- `GET /v1/setup/settings/{id}` → SystemSettingController@show
- `GET /v1/setup/tenants`
- `POST /v1/setup/organizations/{organization}/logo`
- `POST /v1/setup/organizations/{organization}/letterhead`
- `POST /v1/setup/organizations/{organization}/pdf-letterhead`
- `GET /v1/setup/organizations/{organization}/branding`
- `GET /v1/setup/organizations` → OrganizationController@index
- `POST /v1/setup/organizations` → OrganizationController@store
- `GET /v1/setup/organizations/{id}` → OrganizationController@show
- `PUT/PATCH /v1/setup/organizations/{id}` → OrganizationController@update
- `DELETE /v1/setup/organizations/{id}` → OrganizationController@destroy
- `GET /v1/setup/facilities/registry-search`
- `GET /v1/setup/facilities/types`
- `POST /v1/setup/facilities/{facility}/logo`
- `GET /v1/setup/facilities/{facility}/users`
- `GET /v1/setup/facilities` → FacilityController@index
- `POST /v1/setup/facilities` → FacilityController@store
- `GET /v1/setup/facilities/{id}` → FacilityController@show
- `PUT/PATCH /v1/setup/facilities/{id}` → FacilityController@update
- `DELETE /v1/setup/facilities/{id}` → FacilityController@destroy
- `GET /v1/setup/departments` → DepartmentController@index
- `POST /v1/setup/departments` → DepartmentController@store
- `GET /v1/setup/departments/{id}` → DepartmentController@show
- `PUT/PATCH /v1/setup/departments/{id}` → DepartmentController@update
- `DELETE /v1/setup/departments/{id}` → DepartmentController@destroy
- `GET /v1/setup/departments/{id}/employees`
- `GET /v1/setup/destinations/queues`
- `GET /v1/setup/destinations/find`
- `POST /v1/setup/destinations/{id}/deactivate`
- `POST /v1/setup/destinations/{id}/activate`
- `GET /v1/setup/destinations` → ServiceDestinationController@index
- `POST /v1/setup/destinations` → ServiceDestinationController@store
- `GET /v1/setup/destinations/{id}` → ServiceDestinationController@show
- `PUT/PATCH /v1/setup/destinations/{id}` → ServiceDestinationController@update
- `DELETE /v1/setup/destinations/{id}` → ServiceDestinationController@destroy
- `GET /v1/setup/insurance/companies/{id}/schemes`
- `GET /v1/setup/insurance/companies/{id}/pricing`
- `POST /v1/setup/insurance/companies/{id}/pricing`
- `DELETE /v1/setup/insurance/companies/{id}/pricing`
- `GET /v1/setup/insurance/companies/{id}/products`
- `POST /v1/setup/insurance/companies/{id}/products`
- `GET /v1/setup/insurance/companies/{id}/exclusions`
- `POST /v1/setup/insurance/companies/{id}/exclusions`
- `GET /v1/setup/insurance/companies` → InsuranceCompanyController@index
- `POST /v1/setup/insurance/companies` → InsuranceCompanyController@store
- `GET /v1/setup/insurance/companies/{id}` → InsuranceCompanyController@show
- `PUT/PATCH /v1/setup/insurance/companies/{id}` → InsuranceCompanyController@update
- `DELETE /v1/setup/insurance/companies/{id}` → InsuranceCompanyController@destroy
- `GET /v1/setup/insurance/schemes/{id}/exclusions`
- `POST /v1/setup/insurance/schemes/{id}/exclusions`
- `GET /v1/setup/insurance/schemes/{id}/pricing`
- `POST /v1/setup/insurance/schemes/{id}/pricing`
- `GET /v1/setup/insurance/schemes/{id}/procedure-price`
- `GET /v1/setup/insurance/schemes/{id}/procedure-prices`
- `GET /v1/setup/insurance/schemes/{id}/procedures`
- `POST /v1/setup/insurance/schemes/{id}/procedures`
- `GET /v1/setup/insurance/schemes/{id}/exclusions-export`
- `POST /v1/setup/insurance/schemes/{id}/exclusions-import`
- `GET /v1/setup/insurance/schemes/{id}/product-price`
- `GET /v1/setup/insurance/schemes/{id}/products`
- `POST /v1/setup/insurance/schemes/{id}/products`
- `GET /v1/setup/insurance/schemes` → InsuranceSchemeController@index
- `POST /v1/setup/insurance/schemes` → InsuranceSchemeController@store
- `GET /v1/setup/insurance/schemes/{id}` → InsuranceSchemeController@show
- `PUT/PATCH /v1/setup/insurance/schemes/{id}` → InsuranceSchemeController@update
- `DELETE /v1/setup/insurance/schemes/{id}` → InsuranceSchemeController@destroy
- `GET /v1/setup/partnerInstitutions`
- `GET /v1/setup/partners/{id}/staff`
- `GET /v1/setup/partners` → PartnerInstitutionController@index
- `POST /v1/setup/partners` → PartnerInstitutionController@store
- `GET /v1/setup/partners/{id}` → PartnerInstitutionController@show
- `PUT/PATCH /v1/setup/partners/{id}` → PartnerInstitutionController@update
- `DELETE /v1/setup/partners/{id}` → PartnerInstitutionController@destroy
- `GET /v1/setup/partner-staff` → PartnerStaffController@index
- `POST /v1/setup/partner-staff` → PartnerStaffController@store
- `GET /v1/setup/partner-staff/{id}` → PartnerStaffController@show
- `PUT/PATCH /v1/setup/partner-staff/{id}` → PartnerStaffController@update
- `DELETE /v1/setup/partner-staff/{id}` → PartnerStaffController@destroy
- `GET /v1/setup/purpose-of-visits` → PurposeOfVisitController@index
- `POST /v1/setup/purpose-of-visits` → PurposeOfVisitController@store
- `GET /v1/setup/purpose-of-visits/{id}` → PurposeOfVisitController@show
- `PUT/PATCH /v1/setup/purpose-of-visits/{id}` → PurposeOfVisitController@update
- `DELETE /v1/setup/purpose-of-visits/{id}` → PurposeOfVisitController@destroy
- `GET /v1/setup/procedure-categories/{id}/children`
- `GET /v1/setup/procedure-categories` → ProcedureCategoryController@index
- `POST /v1/setup/procedure-categories` → ProcedureCategoryController@store
- `GET /v1/setup/procedure-categories/{id}` → ProcedureCategoryController@show
- `PUT/PATCH /v1/setup/procedure-categories/{id}` → ProcedureCategoryController@update
- `DELETE /v1/setup/procedure-categories/{id}` → ProcedureCategoryController@destroy
- `GET /v1/setup/procedures` → ProcedureController@index
- `POST /v1/setup/procedures` → ProcedureController@store
- `GET /v1/setup/procedures/{id}` → ProcedureController@show
- `PUT/PATCH /v1/setup/procedures/{id}` → ProcedureController@update
- `DELETE /v1/setup/procedures/{id}` → ProcedureController@destroy
- `GET /v1/setup/regions/{id}/counties`
- `GET /v1/setup/regions/{id}/facilities`
- `GET /v1/setup/regions/{region}/users`
- `POST /v1/setup/regions/{region}/users`
- `PUT /v1/setup/regions/{region}/users`
- `DELETE /v1/setup/regions/{region}/users/{user}`
- `GET /v1/setup/users/{user}/regions`
- `POST /v1/setup/users/{user}/regions`
- `PUT /v1/setup/users/{user}/regions`
- `DELETE /v1/setup/users/{user}/regions/{region}`
- `GET /v1/setup/regions` → RegionController@index
- `POST /v1/setup/regions` → RegionController@store
- `GET /v1/setup/regions/{id}` → RegionController@show
- `PUT/PATCH /v1/setup/regions/{id}` → RegionController@update
- `DELETE /v1/setup/regions/{id}` → RegionController@destroy
- `GET /v1/setup/counties` → CountyController@index
- `POST /v1/setup/counties` → CountyController@store
- `GET /v1/setup/counties/{id}` → CountyController@show
- `PUT/PATCH /v1/setup/counties/{id}` → CountyController@update
- `DELETE /v1/setup/counties/{id}` → CountyController@destroy
- `GET /v1/setup/pos/configs/{id}/devices`
- `GET /v1/setup/pos/configs` → PosConfigController@index
- `POST /v1/setup/pos/configs` → PosConfigController@store
- `GET /v1/setup/pos/configs/{id}` → PosConfigController@show
- `PUT/PATCH /v1/setup/pos/configs/{id}` → PosConfigController@update
- `DELETE /v1/setup/pos/configs/{id}` → PosConfigController@destroy
- `POST /v1/setup/pos/devices/{id}/status`
- `POST /v1/setup/pos/devices/{id}/sync`
- `GET /v1/setup/pos/devices` → PosDeviceController@index
- `POST /v1/setup/pos/devices` → PosDeviceController@store
- `GET /v1/setup/pos/devices/{id}` → PosDeviceController@show
- `PUT/PATCH /v1/setup/pos/devices/{id}` → PosDeviceController@update
- `DELETE /v1/setup/pos/devices/{id}` → PosDeviceController@destroy
- `GET /v1/setup/pos/shifts/current` → PosShiftController@current
- `GET /v1/setup/pos/shifts/{id}/reconciliation` → PosShiftController@reconciliation
- `GET /v1/setup/pos/shifts` → PosShiftController@index
- `POST /v1/setup/pos/shifts` → PosShiftController@store
- `GET /v1/setup/pos/shifts/{id}` → PosShiftController@show
- `PUT/PATCH /v1/setup/pos/shifts/{id}` → PosShiftController@update
- `DELETE /v1/setup/pos/shifts/{id}` → PosShiftController@destroy
- `POST /v1/setup/integrations/{id}/test`
- `POST /v1/setup/integrations/{id}/sync`
- `GET /v1/setup/integrations/{id}/logs`
- `GET /v1/setup/integrations` → IntegrationSettingController@index
- `POST /v1/setup/integrations` → IntegrationSettingController@store
- `GET /v1/setup/integrations/{id}` → IntegrationSettingController@show
- `PUT/PATCH /v1/setup/integrations/{id}` → IntegrationSettingController@update
- `DELETE /v1/setup/integrations/{id}` → IntegrationSettingController@destroy
- `GET /v1/setup/themes` → ThemeController@index
- `POST /v1/setup/themes` → ThemeController@store
- `GET /v1/setup/themes/{id}` → ThemeController@show
- `PUT/PATCH /v1/setup/themes/{id}` → ThemeController@update
- `DELETE /v1/setup/themes/{id}` → ThemeController@destroy
- `GET /v1/setup/treatment-actions/{destination}/destinations`
- `POST /v1/setup/treatment-actions/{id}/destinations/sync`
- `GET /v1/setup/treatment-actions` → TreatmentActionController@index
- `POST /v1/setup/treatment-actions` → TreatmentActionController@store
- `GET /v1/setup/treatment-actions/{id}` → TreatmentActionController@show
- `PUT/PATCH /v1/setup/treatment-actions/{id}` → TreatmentActionController@update
- `DELETE /v1/setup/treatment-actions/{id}` → TreatmentActionController@destroy
- `GET /v1/setup/category-filters` → CategoryFilterController@index
- `POST /v1/setup/category-filters` → CategoryFilterController@store
- `GET /v1/setup/category-filters/{id}` → CategoryFilterController@show
- `PUT/PATCH /v1/setup/category-filters/{id}` → CategoryFilterController@update
- `DELETE /v1/setup/category-filters/{id}` → CategoryFilterController@destroy
- `GET /v1/setup/destination-types`
- `GET /v1/setup/rebates` → RebateController@index
- `POST /v1/setup/rebates` → RebateController@store
- `GET /v1/setup/rebates/{id}` → RebateController@show
- `PUT/PATCH /v1/setup/rebates/{id}` → RebateController@update
- `DELETE /v1/setup/rebates/{id}` → RebateController@destroy
- `GET /v1/setup/sample-types` → SampleTypeController@index
- `GET /v1/setup/collection-methods` → SampleCollectionMethodController@index
- `GET /v1/setup/settings/{module}/{setting}` → SettingsController@show
- `POST /v1/setup/settings/{module}/{setting}` → SettingsController@update (needs permission:settings.edit|settings.manage)
- `GET /v1/setup/admin/business/template` → BusinessSetupController@generateCsvTemplate
- `POST /v1/setup/admin/business/validate-csv` → BusinessSetupController@validateCsv
- `POST /v1/setup/admin/business/upload-csv` → BusinessSetupController@uploadCsv
- `POST /v1/setup/admin/business/validate-token` → BusinessSetupController@validateToken
- `POST /v1/setup/admin/business/complete-setup` → BusinessSetupController@completeSetup

### Core service — `sha` endpoints

Area: a {method} name.; catch-all so "general-scheme" is not swallowed as a {method} name.; catch-all so "logs" is not taken for a {method} name.

- `POST /v1/sha/resolve-schemes` → ShaController@resolveSchemes
- `POST /v1/sha/general-scheme` → ShaController@generalScheme
- `GET /v1/sha/logs` → ShaLogController@index
- `GET /v1/sha/logs/accepted-counts` → ShaLogController@acceptedCounts
- `GET /v1/sha/logs/summary` → ShaLogController@summary
- `GET /v1/sha/logs/{id}` → ShaLogController@show
- `POST /v1/sha/claims/attachments` → ShaController@addClaimAttachment
- `GET /v1/sha/claims/attachments/file` → ShaController@downloadClaimAttachment
- `POST /v1/sha/preauths` → ShaController@createPreauth
- `POST /v1/sha/otp-whitelists` → ShaController@requestOtpWhitelist
- `POST /v1/sha/{method}` → ShaController@call

### Core service — `unauthenticated` endpoints

Area: "facility.access",

- `GET /v1/unauthenticated`

### Core service — `user` endpoints

Area: "facility.access",

- `GET /v1/user` → AuthController@user

### Core service — `users` endpoints

Area: User credentials routes; User management routes; User signature routes (digital signatures/approvals); User signatures routes; as reading roles).

- `GET /v1/users/check-username`
- `GET /v1/users` → UserController@index
- `POST /v1/users` → UserController@store
- `GET /v1/users/{id}` → UserController@show
- `PUT/PATCH /v1/users/{id}` → UserController@update
- `DELETE /v1/users/{user}` → UserController@destroy (needs permission:users.delete)
- `POST /v1/users/{user}/assign-roles`
- `DELETE /v1/users/{user}/remove-roles`
- `POST /v1/users/{user}/assign-facilities`
- `DELETE /v1/users/{user}/remove-facilities`
- `GET /v1/users/{user}/facilities`
- `POST /v1/users/{id}/signatures`
- `DELETE /v1/users/{id}/signatures/{signature_id}`
- `PUT /v1/users/{user}/permissions/sync` (needs permission:users.sync_permissions)
- `POST /v1/users/{user}/permissions` (needs permission:users.sync_permissions)
- `DELETE /v1/users/{user}/permissions` (needs permission:users.sync_permissions)
- `GET /v1/users/{user}/permissions`
- `POST /v1/users/{user}/signature`
- `POST /v1/users/{user}/change-password`
- `POST /v1/users/{user}/force-reset-password`
- `GET /v1/users/{user}/login-history`
- `POST /v1/users/{user}/toggle-2fa`
- `GET /v1/users/{user}/api-credentials`
- `POST /v1/users/{user}/api-credentials`
- `DELETE /v1/users/{user}/api-credentials`

### Core service — `verify-email` endpoints

Area: AppServiceProvider::configureRateLimiting(), keyed the same way.

- `POST /v1/verify-email` → AuthController@verifyEmail

### Core service — `verify-reset-otp` endpoints

Area: AppServiceProvider::configureRateLimiting(), keyed the same way.

- `POST /v1/verify-reset-otp` → AuthController@verifyPasswordResetOtp

## Dialysis service API (dialysis-service)

Covers: dialysis patients, schedules, sessions, machines. 114 endpoints.

### Dialysis service — `analytics` endpoints

Area: summary.

- `GET /v1/analytics/summary` → DialysisAnalyticsController@summary
- `GET /v1/analytics/trend` → DialysisAnalyticsController@trend
- `GET /v1/analytics/summary/export` → DialysisAnalyticsController@export

### Dialysis service — `chairs` endpoints

Area: Machine maintenance

- `GET /v1/chairs` → DialysisChairController@index
- `POST /v1/chairs` → DialysisChairController@store
- `GET /v1/chairs/{id}` → DialysisChairController@show
- `PUT/PATCH /v1/chairs/{id}` → DialysisChairController@update
- `DELETE /v1/chairs/{id}` → DialysisChairController@destroy

### Dialysis service — `coagulation-types` endpoints

Area: Machine maintenance

- `GET /v1/coagulation-types` → DialysisCoagulationTypeController@index
- `POST /v1/coagulation-types` → DialysisCoagulationTypeController@store
- `GET /v1/coagulation-types/{id}` → DialysisCoagulationTypeController@show
- `PUT/PATCH /v1/coagulation-types/{id}` → DialysisCoagulationTypeController@update
- `DELETE /v1/coagulation-types/{id}` → DialysisCoagulationTypeController@destroy

### Dialysis service — `deposits` endpoints

Area: Machine maintenance

- `GET /v1/deposits` → DialysisDepositController@index
- `POST /v1/deposits` → DialysisDepositController@store
- `GET /v1/deposits/{id}` → DialysisDepositController@show
- `PUT/PATCH /v1/deposits/{id}` → DialysisDepositController@update
- `DELETE /v1/deposits/{id}` → DialysisDepositController@destroy

### Dialysis service — `dialysis-finance` endpoints

Area: was otherwise unable to reach data their own job requires.

- `GET /v1/dialysis-finance/billable` → DialysisFinanceController@billable

### Dialysis service — `discharges` endpoints

Area: Every discharged session — the Discharged Sessions list.

- `GET /v1/discharges` → DialysisSessionDischargeController@index (needs permission:dialysis.sessions.view|dialysis.manage)

### Dialysis service — `gateway` endpoints

Area: Model Gateway

- `POST /v1/gateway` → ModelGatewayController@handle

### Dialysis service — `health` endpoints

Area: Health check endpoint (public, unversioned)

- `GET /health` → HealthCheckController@check

### Dialysis service — `machine-feed` endpoints

Area: config/hl7.php.

- `POST /v1/machine-feed` → DialysisMachineFeedController@store (needs permission:dialysis.sessions.record_vitals|dialysis.manage)

### Dialysis service — `machines` endpoints

Area: Machine maintenance; Setup

- `GET /v1/machines` → DialysisMachineController@index
- `POST /v1/machines` → DialysisMachineController@store
- `GET /v1/machines/{id}` → DialysisMachineController@show
- `PUT/PATCH /v1/machines/{id}` → DialysisMachineController@update
- `DELETE /v1/machines/{id}` → DialysisMachineController@destroy
- `GET /v1/machines/{machine}/maintenance-logs` → DialysisMachineMaintenanceLogController@index
- `POST /v1/machines/{machine}/maintenance-logs` → DialysisMachineMaintenanceLogController@store (needs permission:dialysis.machines.update|dialysis.setup)
- `PATCH /v1/machines/{machine}/status` → DialysisMachineMaintenanceLogController@updateStatus (needs permission:dialysis.machines.update|dialysis.setup)

### Dialysis service — `patients` endpoints

Area: Idempotent; see DialysisPatientController::ensureEnrolled.

- `POST /v1/patients/ensure-enrolled` → DialysisPatientController@ensureEnrolled (needs permission:dialysis.patients.enroll|dialysis.patients.create|dialysis.manage)
- `GET /v1/patients/{patient}/sessions` → DialysisPatientController@sessions
- `GET /v1/patients` → DialysisPatientController@index
- `POST /v1/patients` → DialysisPatientController@store
- `GET /v1/patients/{id}` → DialysisPatientController@show
- `PUT/PATCH /v1/patients/{id}` → DialysisPatientController@update
- `DELETE /v1/patients/{id}` → DialysisPatientController@destroy

### Dialysis service — `plants` endpoints

Area: Setup

- `GET /v1/plants` → DialysisPlantController@index
- `POST /v1/plants` → DialysisPlantController@store
- `GET /v1/plants/{id}` → DialysisPlantController@show
- `PUT/PATCH /v1/plants/{id}` → DialysisPlantController@update
- `DELETE /v1/plants/{id}` → DialysisPlantController@destroy
- `GET /v1/plants/{plant}/water-tests` → DialysisPlantWaterTestController@index
- `POST /v1/plants/{plant}/water-tests` → DialysisPlantWaterTestController@store (needs permission:dialysis.setup|dialysis.manage)
- `DELETE /v1/plants/{plant}/water-tests/{waterTest}` → DialysisPlantWaterTestController@destroy (needs permission:dialysis.setup|dialysis.manage)

### Dialysis service — `schedule-templates` endpoints

Area: for why the two are separate.; v2's own doc warns about in section 10.

- `POST /v1/schedule-templates/generate` → DialysisScheduleTemplateController@generate (needs permission:dialysis.schedules.create|dialysis.manage)
- `GET /v1/schedule-templates` → DialysisScheduleTemplateController@index
- `POST /v1/schedule-templates` → DialysisScheduleTemplateController@store
- `GET /v1/schedule-templates/{id}` → DialysisScheduleTemplateController@show
- `PUT/PATCH /v1/schedule-templates/{id}` → DialysisScheduleTemplateController@update
- `DELETE /v1/schedule-templates/{id}` → DialysisScheduleTemplateController@destroy

### Dialysis service — `schedules` endpoints

Area: v2's own doc warns about in section 10.

- `POST /v1/schedules/{schedule}/check-in` → DialysisScheduleController@checkIn (needs permission:dialysis.schedules.update|dialysis.manage)
- `POST /v1/schedules/{schedule}/cancel` → DialysisScheduleController@cancel (needs permission:dialysis.schedules.cancel|dialysis.schedules.update)
- `GET /v1/schedules` → DialysisScheduleController@index
- `POST /v1/schedules` → DialysisScheduleController@store
- `GET /v1/schedules/{id}` → DialysisScheduleController@show
- `PUT/PATCH /v1/schedules/{id}` → DialysisScheduleController@update
- `DELETE /v1/schedules/{id}` → DialysisScheduleController@destroy

### Dialysis service — `sessions` endpoints

Area: Every discharged session — the Discharged Sessions list.; Fluid balance (parity gap 4) - distinct from the vitals chart.; No delete: what was given is a clinical record.; Session clinical documentation; Sessions; The dialysate prescription — bath composition, UF goal and target weight.

- `GET /v1/sessions/today` → DialysisSessionController@today
- `POST /v1/sessions/{session}/start` → DialysisSessionController@start (needs permission:dialysis.sessions.start|dialysis.manage)
- `POST /v1/sessions/{session}/check-in` → DialysisSessionController@checkIn (needs permission:dialysis.sessions.start|dialysis.sessions.update|dialysis.manage)
- `GET /v1/sessions/{session}/stages` → DialysisSessionStageController@index
- `POST /v1/sessions/{session}/stages/{stage}/complete` → DialysisSessionStageController@complete (needs permission:dialysis.sessions.update|dialysis.sessions.record_assessment|dialysis.manage)
- `DELETE /v1/sessions/{session}/stages/{stage}` → DialysisSessionStageController@reopen (needs permission:dialysis.sessions.update|dialysis.manage)
- `POST /v1/sessions/{session}/dialysate-prescription` → DialysisSessionController@storeDialysatePrescription (needs permission:dialysis.sessions.update|dialysis.sessions.record_assessment|dialysis.manage)
- `POST /v1/sessions/{session}/complete` → DialysisSessionController@complete (needs permission:dialysis.sessions.complete|dialysis.manage)
- `GET /v1/sessions/{session}/discharge` → DialysisSessionDischargeController@show
- `POST /v1/sessions/{session}/discharge` → DialysisSessionDischargeController@store (needs permission:dialysis.sessions.complete|dialysis.manage)
- `POST /v1/sessions/{session}/cancel` → DialysisSessionController@cancel (needs permission:dialysis.sessions.cancel|dialysis.manage)
- `GET /v1/sessions/{session}/vitals` → DialysisSessionController@listVitals
- `POST /v1/sessions/{session}/vitals` → DialysisSessionController@addVital (needs permission:dialysis.sessions.record_vitals|dialysis.manage)
- `GET /v1/sessions/{session}/fluid-balances` → DialysisSessionController@fluidBalances
- `POST /v1/sessions/{session}/fluid-balances` → DialysisSessionController@storeFluidBalance (needs permission:dialysis.sessions.record_vitals|dialysis.manage)
- `DELETE /v1/sessions/{session}/fluid-balances/{entryId}` → DialysisSessionController@destroyFluidBalance (needs permission:dialysis.sessions.record_vitals|dialysis.manage)
- `GET /v1/sessions/{session}/assessments` → DialysisAssessmentController@get
- `POST /v1/sessions/{session}/pre-dialysis` → DialysisAssessmentController@storePreDialysis (needs permission:dialysis.sessions.record_assessment|dialysis.manage)
- `POST /v1/sessions/{session}/assessments/vascular` → DialysisAssessmentController@storeVascular (needs permission:dialysis.sessions.record_assessment|dialysis.manage)
- `POST /v1/sessions/{session}/assessments/machine` → DialysisAssessmentController@storeMachine (needs permission:dialysis.sessions.record_assessment|dialysis.manage)
- `POST /v1/sessions/{session}/assessments/patient` → DialysisAssessmentController@storePatient (needs permission:dialysis.sessions.record_assessment|dialysis.manage)
- `POST /v1/sessions/{session}/assessments/heparinization` → DialysisAssessmentController@storeHeparinization (needs permission:dialysis.sessions.record_assessment|dialysis.manage)
- `GET /v1/sessions/{session}/medication-prescriptions` → DialysisMedicationPrescriptionController@index
- `POST /v1/sessions/{session}/medication-prescriptions` → DialysisMedicationPrescriptionController@store (needs permission:dialysis.sessions.add_medication|dialysis.manage)
- `GET /v1/sessions/{session}/medications` → DialysisSessionMedicationController@index
- `POST /v1/sessions/{session}/medications` → DialysisSessionMedicationController@store (needs permission:dialysis.sessions.add_medication|dialysis.manage)
- `GET /v1/sessions/{session}/consumables` → DialysisSessionConsumableController@index
- `POST /v1/sessions/{session}/consumables` → DialysisSessionConsumableController@store (needs permission:dialysis.sessions.record_consumables|dialysis.manage)
- `DELETE /v1/sessions/{session}/consumables/{consumableId}` → DialysisSessionConsumableController@destroy (needs permission:dialysis.sessions.record_consumables|dialysis.manage)
- `GET /v1/sessions/{session}/notes` → DialysisSessionNoteController@index
- `POST /v1/sessions/{session}/notes` → DialysisSessionNoteController@store
- `DELETE /v1/sessions/{session}/notes/{noteId}` → DialysisSessionNoteController@destroy
- `GET /v1/sessions/{session}/charge-sheet` → DialysisSessionController@chargeSheet
- `GET /v1/sessions` → DialysisSessionController@index
- `POST /v1/sessions` → DialysisSessionController@store
- `GET /v1/sessions/{id}` → DialysisSessionController@show
- `PUT/PATCH /v1/sessions/{id}` → DialysisSessionController@update
- `DELETE /v1/sessions/{id}` → DialysisSessionController@destroy
- `GET /v1/sessions/{session}/billing-data` → DialysisFinanceController@billingData
- `POST /v1/sessions/{session}/mark-billed` → DialysisFinanceController@markBilled

### Dialysis service — `vascular-accesses` endpoints

Area: Machine maintenance

- `GET /v1/vascular-accesses` → DialysisVascularAccessController@index
- `POST /v1/vascular-accesses` → DialysisVascularAccessController@store
- `GET /v1/vascular-accesses/{id}` → DialysisVascularAccessController@show
- `PUT/PATCH /v1/vascular-accesses/{id}` → DialysisVascularAccessController@update
- `DELETE /v1/vascular-accesses/{id}` → DialysisVascularAccessController@destroy

### Dialysis service — `ward-types` endpoints

Area: Setup

- `GET /v1/ward-types` → DialysisWardTypeController@index
- `POST /v1/ward-types` → DialysisWardTypeController@store
- `GET /v1/ward-types/{id}` → DialysisWardTypeController@show
- `PUT/PATCH /v1/ward-types/{id}` → DialysisWardTypeController@update
- `DELETE /v1/ward-types/{id}` → DialysisWardTypeController@destroy

### Dialysis service — `wards` endpoints

Area: Setup

- `GET /v1/wards` → DialysisWardController@index
- `POST /v1/wards` → DialysisWardController@store
- `GET /v1/wards/{id}` → DialysisWardController@show
- `PUT/PATCH /v1/wards/{id}` → DialysisWardController@update
- `DELETE /v1/wards/{id}` → DialysisWardController@destroy

## Finance service API (finance-service)

Covers: billing, cash workbench, invoices, receipts, credit notes, waivers, insurance. 466 endpoints.

### Finance service — `accounting-api-reference` endpoints

Area: Configuration

- `GET /v1/accounting-api-reference` → AccountingApiReferenceController@index
- `POST /v1/accounting-api-reference` → AccountingApiReferenceController@store
- `GET /v1/accounting-api-reference/{id}` → AccountingApiReferenceController@show
- `PUT/PATCH /v1/accounting-api-reference/{id}` → AccountingApiReferenceController@update
- `DELETE /v1/accounting-api-reference/{id}` → AccountingApiReferenceController@destroy

### Finance service — `accounting-api-reference-stats` endpoints

Area: Configuration

- `GET /v1/accounting-api-reference-stats` → AccountingApiReferenceStatsController@index
- `POST /v1/accounting-api-reference-stats` → AccountingApiReferenceStatsController@store
- `GET /v1/accounting-api-reference-stats/{id}` → AccountingApiReferenceStatsController@show
- `PUT/PATCH /v1/accounting-api-reference-stats/{id}` → AccountingApiReferenceStatsController@update
- `DELETE /v1/accounting-api-reference-stats/{id}` → AccountingApiReferenceStatsController@destroy

### Finance service — `accounting-sync` endpoints

Area: the finance failure screen (see, resend, mark resolved).; the frontend top-bar indicator polls this.

- `GET /v1/accounting-sync/status` → AccountingSyncStatusController@status
- `POST /v1/accounting-sync/failures/seen` → AccountingSyncStatusController@seen
- `GET /v1/accounting-sync/failures` (needs permission:accounts.reconcile|payments.reconcile|finance.manage)
- `GET /v1/accounting-sync/failures/{id}` (needs permission:accounts.reconcile|payments.reconcile|finance.manage)
- `POST /v1/accounting-sync/failures/{id}/resend` (needs permission:accounts.reconcile|payments.reconcile|finance.manage)
- `POST /v1/accounting-sync/failures/{id}/resolve` (needs permission:accounts.reconcile|payments.reconcile|finance.manage)

### Finance service — `bank` endpoints

Area: Configuration

- `GET /v1/bank` → BankController@index
- `POST /v1/bank` → BankController@store
- `GET /v1/bank/{id}` → BankController@show
- `PUT/PATCH /v1/bank/{id}` → BankController@update
- `DELETE /v1/bank/{id}` → BankController@destroy

### Finance service — `bank-account` endpoints

Area: Configuration

- `GET /v1/bank-account` → BankAccountController@index
- `POST /v1/bank-account` → BankAccountController@store
- `GET /v1/bank-account/{id}` → BankAccountController@show
- `PUT/PATCH /v1/bank-account/{id}` → BankAccountController@update
- `DELETE /v1/bank-account/{id}` → BankAccountController@destroy

### Finance service — `banking` endpoints

Area: ===== PHASE 2: BANKING =====

- `POST /v1/banking/import-feed` → BankingController@importBankFeed (needs permission:payments.reconcile|accounts.reconcile|finance.manage)
- `GET /v1/banking/unreconciled` → BankingController@getMismatchedTransactions
- `POST /v1/banking/reconcile` → BankingController@reconcile (needs permission:payments.reconcile|accounts.reconcile|finance.manage)
- `GET /v1/banking/reconciliation-summary` → BankingController@getReconciliationReport
- `POST /v1/banking/flag-for-review` → BankingController@flagForReview (needs permission:payments.reconcile|accounts.reconcile|finance.manage)

### Finance service — `banking-cheque` endpoints

Area: Configuration

- `GET /v1/banking-cheque` → BankingChequeController@index
- `POST /v1/banking-cheque` → BankingChequeController@store
- `GET /v1/banking-cheque/{id}` → BankingChequeController@show
- `PUT/PATCH /v1/banking-cheque/{id}` → BankingChequeController@update
- `DELETE /v1/banking-cheque/{id}` → BankingChequeController@destroy

### Finance service — `billing` endpoints

Area: ===== BILLING OPERATIONS =====; directly, so settlement genuinely passes through Finance first.

- `POST /v1/billing/store` → PatientBillsController@store
- `POST /v1/billing/pos-payment` → PatientBillsController@posPayment
- `GET /v1/billing/pending-pharmacy-sales` → PatientBillsController@pendingPharmacySales (needs permission:pos.sales.receive_payment|finance.manage)
- `POST /v1/billing/pending-pharmacy-sales/{saleId}/process-payment` → PatientBillsController@processPendingPharmacySale (needs permission:pos.sales.receive_payment|finance.manage)
- `POST /v1/billing/cash/{key}` → CashierController@store (needs permission:payments.process|finance.process_payments)
- `GET /v1/billing/payment/{id}` → PatientBillsController@showPaymentDetails
- `POST /v1/billing/prepare` → BillingController@prepare
- `POST /v1/billing/settle` → BillingController@settle (needs permission:payments.process|finance.process_payments)
- `POST /v1/billing/{billId}/cancel` → BillingController@cancel (needs permission:invoices.void|billing.manage|finance.manage)
- `POST /v1/billing/{billId}/undo-cancel` → BillingController@undoCancel (needs permission:invoices.void|billing.manage|finance.manage)
- `GET /v1/billing/pending` → BillingController@getPending
- `GET /v1/billing/check-balance` → BillingController@checkBalance

### Finance service — `billings` endpoints

Area: can hit them directly.

- `GET /v1/billings` → BillingController@index
- `POST /v1/billings` → BillingController@store
- `GET /v1/billings/{id}` → BillingController@show
- `PUT/PATCH /v1/billings/{id}` → BillingController@update
- `DELETE /v1/billings/{id}` → BillingController@destroy

### Finance service — `cash` endpoints

Area: Adjustments; Called by reception-service after patient creation — idempotent and harmless (creates empty account only).; Cash-prefixed health check endpoint (versioned); Cash-specific invoice routes (used by the Create Cash Invoice UI); Configuration; Insurance Management

- `POST /v1/cash/patients/{patientId}/ensure-account` → CashSelfAccountController@ensureAccount
- `GET /v1/cash/health` → CashHealthController@index
- `GET /v1/cash/invoices` → InvoiceController@index
- `POST /v1/cash/invoices` → InvoiceController@store
- `GET /v1/cash/invoices/{id}` → InvoiceController@show
- `PUT/PATCH /v1/cash/invoices/{id}` → InvoiceController@update
- `DELETE /v1/cash/invoices/{id}` → InvoiceController@destroy
- `GET /v1/cash/invoices` → CashInvoiceController@index
- `POST /v1/cash/invoices` → CashInvoiceController@store
- `GET /v1/cash/invoices/{id}` → CashInvoiceController@show
- `PUT/PATCH /v1/cash/invoices/{id}` → CashInvoiceController@update
- `DELETE /v1/cash/invoices/{id}` → CashInvoiceController@destroy
- `GET /v1/cash/invoice-items` → CashInvoiceItemController@index
- `POST /v1/cash/invoice-items` → CashInvoiceItemController@store
- `GET /v1/cash/invoice-items/{id}` → CashInvoiceItemController@show
- `PUT/PATCH /v1/cash/invoice-items/{id}` → CashInvoiceItemController@update
- `DELETE /v1/cash/invoice-items/{id}` → CashInvoiceItemController@destroy
- `POST /v1/cash/invoices/{invoice}/payments` → CashInvoiceController@addPayment
- `POST /v1/cash/invoices/{invoice}/items` → CashInvoiceController@addItem
- `GET /v1/cash/payments` → CashPaymentController@index
- `POST /v1/cash/payments` → CashPaymentController@store
- `GET /v1/cash/payments/{id}` → CashPaymentController@show
- `PUT/PATCH /v1/cash/payments/{id}` → CashPaymentController@update
- `DELETE /v1/cash/payments/{id}` → CashPaymentController@destroy
- `GET /v1/cash/evaluation-payments` → CashEvaluationPaymentController@index
- `POST /v1/cash/evaluation-payments` → CashEvaluationPaymentController@store
- `GET /v1/cash/evaluation-payments/{id}` → CashEvaluationPaymentController@show
- `PUT/PATCH /v1/cash/evaluation-payments/{id}` → CashEvaluationPaymentController@update
- `DELETE /v1/cash/evaluation-payments/{id}` → CashEvaluationPaymentController@destroy
- `POST /v1/cash/payments/{payment}/allocate` → CashPaymentController@allocate
- `GET /v1/cash/patient-accounts` → CashPatientAccountController@index
- `POST /v1/cash/patient-accounts` → CashPatientAccountController@store
- `GET /v1/cash/patient-accounts/{id}` → CashPatientAccountController@show
- `PUT/PATCH /v1/cash/patient-accounts/{id}` → CashPatientAccountController@update
- `DELETE /v1/cash/patient-accounts/{id}` → CashPatientAccountController@destroy
- `POST /v1/cash/patient-accounts/{patientAccount}/deposit` → CashPatientAccountController@deposit
- `POST /v1/cash/patient-accounts/{patientAccount}/withdraw` → CashPatientAccountController@withdraw (needs permission:deposits.refund|finance.manage)
- `GET /v1/cash/patient-accounts/{patientAccount}/transactions` → CashPatientAccountController@transactions
- `GET /v1/cash/self-accounts` → CashSelfAccountController@index
- `GET /v1/cash/self-accounts/{patient}/overview` → CashSelfAccountController@overview
- `GET /v1/cash/self-accounts/{patient}/deposits` → CashSelfAccountController@deposits
- `POST /v1/cash/self-accounts/{patient}/deposits` → CashSelfAccountController@storeDeposit
- `GET /v1/cash/self-accounts/{patient}/withdrawals` → CashSelfAccountController@withdrawals
- `POST /v1/cash/self-accounts/{patient}/withdrawals` → CashSelfAccountController@storeWithdrawal (needs permission:deposits.refund|finance.manage)
- `GET /v1/cash/self-accounts/{patient}/withdrawals/{withdrawal}/approve` → CashSelfAccountController@approveWithdrawal (needs permission:deposits.refund|finance.manage)
- `GET /v1/cash/self-accounts/{patient}/cash-payments` → CashSelfAccountController@cashPayments
- `GET /v1/cash/self-accounts/{patient}/invoice-payments` → CashSelfAccountController@invoicePayments
- `GET /v1/cash/self-accounts/{patient}/gift-cards` → CashSelfAccountController@giftCards
- `GET /v1/cash/self-accounts/{patient}/vouchers` → CashSelfAccountController@vouchers
- `GET /v1/cash/self-accounts/{patient}/deposit-requests` → CashSelfAccountController@depositRequests
- `POST /v1/cash/self-accounts/{patient}/deposit-requests` → CashSelfAccountController@storeDepositRequest
- `POST /v1/cash/self-accounts/{patient}/deposit-requests/{depositRequest}/resolve` → CashSelfAccountController@resolveDepositRequest
- `PATCH /v1/cash/self-accounts/{patient}/deposit-requests/{depositRequest}` → CashSelfAccountController@updateDepositRequestAmount
- `GET /v1/cash/insurance-companies` → CashInsuranceCompanyController@index
- `POST /v1/cash/insurance-companies` → CashInsuranceCompanyController@store
- `GET /v1/cash/insurance-companies/{id}` → CashInsuranceCompanyController@show
- `PUT/PATCH /v1/cash/insurance-companies/{id}` → CashInsuranceCompanyController@update
- `DELETE /v1/cash/insurance-companies/{id}` → CashInsuranceCompanyController@destroy
- `GET /v1/cash/insurance-schemes` → CashInsuranceSchemeController@index
- `POST /v1/cash/insurance-schemes` → CashInsuranceSchemeController@store
- `GET /v1/cash/insurance-schemes/{id}` → CashInsuranceSchemeController@show
- `PUT/PATCH /v1/cash/insurance-schemes/{id}` → CashInsuranceSchemeController@update
- `DELETE /v1/cash/insurance-schemes/{id}` → CashInsuranceSchemeController@destroy
- `GET /v1/cash/payment-modes` → CashPaymentModeController@index
- `GET /v1/cash/payment-modes/{id}` → CashPaymentModeController@show
- `GET /v1/cash/service-charges` → CashServiceChargeController@index
- `POST /v1/cash/service-charges` → CashServiceChargeController@store
- `GET /v1/cash/service-charges/{id}` → CashServiceChargeController@show
- `PUT/PATCH /v1/cash/service-charges/{id}` → CashServiceChargeController@update
- `DELETE /v1/cash/service-charges/{id}` → CashServiceChargeController@destroy
- `GET /v1/cash/credit-notes` → CashCreditNoteController@index
- `POST /v1/cash/credit-notes` → CashCreditNoteController@store
- `GET /v1/cash/credit-notes/{id}` → CashCreditNoteController@show
- `PUT/PATCH /v1/cash/credit-notes/{id}` → CashCreditNoteController@update
- `DELETE /v1/cash/credit-notes/{id}` → CashCreditNoteController@destroy
- `GET /v1/cash/waivers` → CashWaiverController@index
- `POST /v1/cash/waivers` → CashWaiverController@store
- `GET /v1/cash/waivers/{id}` → CashWaiverController@show
- `PUT/PATCH /v1/cash/waivers/{id}` → CashWaiverController@update
- `DELETE /v1/cash/waivers/{id}` → CashWaiverController@destroy
- `GET /v1/cash/reports/revenue` → CashReportController@revenue
- `GET /v1/cash/reports/outstanding-invoices` → CashReportController@outstandingInvoices
- `GET /v1/cash/reports/reconciliation` → CashReportController@reconciliation
- `POST /v1/cash/integrations/mpesa/callback` → CashIntegrationController@mpesaCallback
- `POST /v1/cash/integrations/pesapal/callback` → CashIntegrationController@pesapalCallback
- `GET /v1/cash/integrations/sync-status` → CashIntegrationController@syncStatus

### Finance service — `cash-payments` endpoints

Area: Standalone payment endpoints (without namespace prefix)

- `GET /v1/cash-payments` → StandaloneCashPaymentController@index
- `POST /v1/cash-payments` → StandaloneCashPaymentController@store
- `GET /v1/cash-payments/receipts` → StandaloneCashPaymentController@receipts
- `GET /v1/cash-payments/sales-summary` → StandaloneCashPaymentController@salesSummary
- `POST /v1/cash-payments/receipt/{receipt}/cancel` → StandaloneCashPaymentController@cancelReceipt (needs permission:receipts.void|payments.reverse|finance.manage)
- `POST /v1/cash-payments/receipt/{receipt}/credit-note` → StandaloneCashPaymentController@issueCreditNote (needs permission:credit_notes.create|receipts.void|finance.manage)
- `POST /v1/cash-payments/credit-note/{creditNote}/approve` → StandaloneCashPaymentController@approveCreditNote (needs permission:credit_notes.approve|finance.manage)

### Finance service — `company-statements` endpoints

Area: ===== PHASE 2: COMPANY STATEMENTS =====

- `GET /v1/company-statements/companies` → CompanyStatementsController@companies
- `GET /v1/company-statements/companies/export` → CompanyStatementsController@companiesExport
- `POST /v1/company-statements/generate` → CompanyStatementsController@generateStatement
- `GET /v1/company-statements/download` → CompanyStatementsController@downloadStatement
- `GET /v1/company-statements/outstanding` → CompanyStatementsController@getOutstandingBalance
- `GET /v1/company-statements/payments` → CompanyStatementsController@getPaymentHistory
- `GET /v1/company-statements/aging` → CompanyStatementsController@agingReport

### Finance service — `credit-note` endpoints

Area: ===== PRINTING =====

- `POST /v1/credit-note/{creditNote}/approve` → StandaloneCashPaymentController@approveCreditNote (needs permission:credit_notes.approve|finance.manage)

### Finance service — `credit-note-transfers` endpoints

Area: money is not the one that asks for it.

- `GET /v1/credit-note-transfers` → CreditNoteAccountTransferController@index (needs permission:credit_notes.view|credit_notes.approve|finance.manage)
- `POST /v1/credit-note-transfers/{transfer}/approve` → CreditNoteAccountTransferController@approve (needs permission:credit_notes.approve|finance.manage)
- `POST /v1/credit-note-transfers/{transfer}/reject` → CreditNoteAccountTransferController@reject (needs permission:credit_notes.approve|finance.manage)

### Finance service — `credit-notes` endpoints

Area: money is not the one that asks for it.; needs to precede the apiResource here too, not just down there.; {creditNote}/show route swallows this as creditNote="refunded-summary".

- `GET /v1/credit-notes/refunded-summary` → CreditNoteController@refundedSummary
- `GET /v1/credit-notes` → CreditNoteController@index
- `POST /v1/credit-notes` → CreditNoteController@store
- `GET /v1/credit-notes/{id}` → CreditNoteController@show
- `PUT/PATCH /v1/credit-notes/{id}` → CreditNoteController@update
- `DELETE /v1/credit-notes/{id}` → CreditNoteController@destroy
- `POST /v1/credit-notes/{creditNote}/approve` → CreditNoteController@approve (needs permission:credit_notes.approve|finance.manage)
- `POST /v1/credit-notes/{creditNote}/cancel` → CreditNoteController@cancel (needs permission:credit_notes.void|finance.manage)
- `POST /v1/credit-notes/{creditNote}/update-status` → CreditNoteController@updateStatus (needs permission:credit_notes.approve|credit_notes.void|finance.manage)
- `GET /v1/credit-notes/refunded-summary` → CreditNoteController@refundedSummary
- `GET /v1/credit-notes` → CreditNoteController@index
- `POST /v1/credit-notes` → CreditNoteController@store
- `GET /v1/credit-notes/{id}` → CreditNoteController@show
- `PUT/PATCH /v1/credit-notes/{id}` → CreditNoteController@update
- `DELETE /v1/credit-notes/{id}` → CreditNoteController@destroy
- `POST /v1/credit-notes/{creditNote}/refund` → CreditNoteController@refund (needs permission:payments.reverse|deposits.refund|finance.manage)
- `POST /v1/credit-notes/{creditNote}/to-account` → CreditNoteController@toAccount (needs permission:credit_notes.apply|deposits.apply|finance.manage)
- `POST /v1/credit-notes/{creditNote}/update-status` → CreditNoteController@updateStatus (needs permission:credit_notes.approve|credit_notes.void|finance.manage)

### Finance service — `dashboard` endpoints

Area: on DashboardMetricsController but never routed until now.

- `GET /v1/dashboard/finance-summary` → DashboardMetricsController@financeSummary

### Finance service — `dialysis-finance` endpoints

Area: consumable/medication data over the gateway plus the billed_at mutation.

- `GET /v1/dialysis-finance/billable` → DialysisFinanceController@billable

### Finance service — `dialysis-sessions` endpoints

Area: consumable/medication data over the gateway plus the billed_at mutation.

- `GET /v1/dialysis-sessions/{session}/charge-sheet` → DialysisFinanceController@chargeSheet
- `POST /v1/dialysis-sessions/{session}/mark-billed` → DialysisFinanceController@markBilled

### Finance service — `dispatch` endpoints

Area: Configuration

- `GET /v1/dispatch` → DispatchController@index
- `POST /v1/dispatch` → DispatchController@store
- `GET /v1/dispatch/{id}` → DispatchController@show
- `PUT/PATCH /v1/dispatch/{id}` → DispatchController@update
- `DELETE /v1/dispatch/{id}` → DispatchController@destroy

### Finance service — `dynamic-item-sent` endpoints

Area: Configuration

- `GET /v1/dynamic-item-sent` → DynamicItemSentController@index
- `POST /v1/dynamic-item-sent` → DynamicItemSentController@store
- `GET /v1/dynamic-item-sent/{id}` → DynamicItemSentController@show
- `PUT/PATCH /v1/dynamic-item-sent/{id}` → DynamicItemSentController@update
- `DELETE /v1/dynamic-item-sent/{id}` → DynamicItemSentController@destroy

### Finance service — `evaluation-payments` endpoints

Area: Payments Management

- `GET /v1/evaluation-payments` → EvaluationPaymentController@index
- `POST /v1/evaluation-payments` → EvaluationPaymentController@store
- `GET /v1/evaluation-payments/{id}` → EvaluationPaymentController@show
- `PUT/PATCH /v1/evaluation-payments/{id}` → EvaluationPaymentController@update
- `DELETE /v1/evaluation-payments/{id}` → EvaluationPaymentController@destroy

### Finance service — `evaluation-payments-details` endpoints

Area: Configuration

- `GET /v1/evaluation-payments-details` → EvaluationPaymentsDetailsController@index
- `POST /v1/evaluation-payments-details` → EvaluationPaymentsDetailsController@store
- `GET /v1/evaluation-payments-details/{id}` → EvaluationPaymentsDetailsController@show
- `PUT/PATCH /v1/evaluation-payments-details/{id}` → EvaluationPaymentsDetailsController@update
- `DELETE /v1/evaluation-payments-details/{id}` → EvaluationPaymentsDetailsController@destroy

### Finance service — `finance` endpoints

Area: SHA claim builds — the Claim builds tab (ShaClaimBuilds).

- `GET /v1/finance/invoices/{key}/action/{action}` → InsuranceInvoiceController@action
- `POST /v1/finance/invoices/{key}/store` → InsuranceInvoiceController@storeAction
- `POST /v1/finance/invoices/split-bill/{visitId}` → InsuranceInvoiceController@splitBill
- `GET /v1/finance/invoices/pending/action/split` → InsuranceInvoiceController@split

### Finance service — `finance-account-group` endpoints

Area: Configuration

- `GET /v1/finance-account-group` → FinanceAccountGroupController@index
- `POST /v1/finance-account-group` → FinanceAccountGroupController@store
- `GET /v1/finance-account-group/{id}` → FinanceAccountGroupController@show
- `PUT/PATCH /v1/finance-account-group/{id}` → FinanceAccountGroupController@update
- `DELETE /v1/finance-account-group/{id}` → FinanceAccountGroupController@destroy

### Finance service — `finance-account-type` endpoints

Area: Configuration

- `GET /v1/finance-account-type` → FinanceAccountTypeController@index
- `POST /v1/finance-account-type` → FinanceAccountTypeController@store
- `GET /v1/finance-account-type/{id}` → FinanceAccountTypeController@show
- `PUT/PATCH /v1/finance-account-type/{id}` → FinanceAccountTypeController@update
- `DELETE /v1/finance-account-type/{id}` → FinanceAccountTypeController@destroy

### Finance service — `finance-evaluation-insurance-payments` endpoints

Area: Configuration

- `GET /v1/finance-evaluation-insurance-payments` → FinanceEvaluationInsurancePaymentsController@index
- `POST /v1/finance-evaluation-insurance-payments` → FinanceEvaluationInsurancePaymentsController@store
- `GET /v1/finance-evaluation-insurance-payments/{id}` → FinanceEvaluationInsurancePaymentsController@show
- `PUT/PATCH /v1/finance-evaluation-insurance-payments/{id}` → FinanceEvaluationInsurancePaymentsController@update
- `DELETE /v1/finance-evaluation-insurance-payments/{id}` → FinanceEvaluationInsurancePaymentsController@destroy

### Finance service — `finance-gl-accounts` endpoints

Area: Configuration

- `GET /v1/finance-gl-accounts` → FinanceGlAccountsController@index
- `POST /v1/finance-gl-accounts` → FinanceGlAccountsController@store
- `GET /v1/finance-gl-accounts/{id}` → FinanceGlAccountsController@show
- `PUT/PATCH /v1/finance-gl-accounts/{id}` → FinanceGlAccountsController@update
- `DELETE /v1/finance-gl-accounts/{id}` → FinanceGlAccountsController@destroy

### Finance service — `finance-transactions` endpoints

Area: Configuration

- `GET /v1/finance-transactions` → FinanceTransactionsController@index
- `POST /v1/finance-transactions` → FinanceTransactionsController@store
- `GET /v1/finance-transactions/{id}` → FinanceTransactionsController@show
- `PUT/PATCH /v1/finance-transactions/{id}` → FinanceTransactionsController@update
- `DELETE /v1/finance-transactions/{id}` → FinanceTransactionsController@destroy

### Finance service — `gateway` endpoints

Area: Model Gateway — cross-tenant read / seed / copy (superadmin only)

- `POST /v1/gateway` → ModelGatewayController@handle

### Finance service — `gl-accounts` endpoints

Area: ===== PHASE 2: GL ACCOUNTS =====

- `GET /v1/gl-accounts` → GLAccountController@listGLAccounts
- `GET /v1/gl-accounts/{accountId}` → GLAccountController@show
- `POST /v1/gl-accounts/mapping` → GLAccountController@createMapping (needs permission:accounts.create|accounts.update|finance.manage)
- `GET /v1/gl-accounts/{accountId}/transactions` → GLAccountController@getTransactions
- `POST /v1/gl-accounts/{accountId}/reconcile` → GLAccountController@reconcile (needs permission:accounts.reconcile|payments.reconcile|finance.manage)
- `GET /v1/gl-accounts/balances` → GLAccountController@getBalances

### Finance service — `health` endpoints

Area: Health check endpoint (unversioned, public)

- `GET /health` → HealthController@index

### Finance service — `holding-stock` endpoints

Area: Configuration

- `GET /v1/holding-stock` → HoldingStockController@index
- `POST /v1/holding-stock` → HoldingStockController@store
- `GET /v1/holding-stock/{id}` → HoldingStockController@show
- `PUT/PATCH /v1/holding-stock/{id}` → HoldingStockController@update
- `DELETE /v1/holding-stock/{id}` → HoldingStockController@destroy

### Finance service — `insurance` endpoints

Area: ===== PHASE 2: INSURANCE INTEGRATION =====

- `POST /v1/insurance/submit-claim` → InsuranceIntegrationController@submitClaim (needs permission:insurance_claims.submit|finance.manage)
- `GET /v1/insurance/claims` → InsuranceIntegrationController@getClaimHistory
- `GET /v1/insurance/claim/{claimId}/status` → InsuranceIntegrationController@getClaimStatus
- `POST /v1/insurance/claim/{claimId}/approve` → InsuranceIntegrationController@updateApproval (needs permission:insurance_claims.approve|finance.manage)
- `POST /v1/insurance/claim/{claimId}/reject` → InsuranceIntegrationController@rejectClaim (needs permission:insurance_claims.reject|finance.manage)

### Finance service — `insurance-cheque` endpoints

Area: Configuration

- `GET /v1/insurance-cheque` → InsuranceChequeController@index
- `POST /v1/insurance-cheque` → InsuranceChequeController@store
- `GET /v1/insurance-cheque/{id}` → InsuranceChequeController@show
- `PUT/PATCH /v1/insurance-cheque/{id}` → InsuranceChequeController@update
- `DELETE /v1/insurance-cheque/{id}` → InsuranceChequeController@destroy

### Finance service — `insurance-companies` endpoints

Area: Insurance Management

- `GET /v1/insurance-companies` → InsuranceCompanyController@index
- `POST /v1/insurance-companies` → InsuranceCompanyController@store
- `GET /v1/insurance-companies/{id}` → InsuranceCompanyController@show
- `PUT/PATCH /v1/insurance-companies/{id}` → InsuranceCompanyController@update
- `DELETE /v1/insurance-companies/{id}` → InsuranceCompanyController@destroy

### Finance service — `insurance-invoices` endpoints

Area: declared first or they fall through to show() with that segment as the id.

- `GET /v1/insurance-invoices/graph` → InsuranceInvoiceController@graph
- `GET /v1/insurance-invoices/visit/{visitId}` → InsuranceInvoiceController@forVisit
- `GET /v1/insurance-invoices` → InsuranceInvoiceController@index
- `POST /v1/insurance-invoices` → InsuranceInvoiceController@store
- `GET /v1/insurance-invoices/{id}` → InsuranceInvoiceController@show
- `PUT/PATCH /v1/insurance-invoices/{id}` → InsuranceInvoiceController@update
- `DELETE /v1/insurance-invoices/{id}` → InsuranceInvoiceController@destroy

### Finance service — `insurance-schemes` endpoints

Area: Insurance Management

- `GET /v1/insurance-schemes` → InsuranceSchemeController@index
- `POST /v1/insurance-schemes` → InsuranceSchemeController@store
- `GET /v1/insurance-schemes/{id}` → InsuranceSchemeController@show
- `PUT/PATCH /v1/insurance-schemes/{id}` → InsuranceSchemeController@update
- `DELETE /v1/insurance-schemes/{id}` → InsuranceSchemeController@destroy

### Finance service — `integrations` endpoints

Area: factory entry rather than a new controller method.; shapes, so each has its own route that delegates to handleCallback().

- `POST /v1/integrations/{provider}/initiate` → PaymentIntegrationController@initiate
- `POST /v1/integrations/mpesa/callback` → PaymentIntegrationController@mpesaCallback
- `POST /v1/integrations/pesapal/callback` → PaymentIntegrationController@pesapalCallback
- `GET /v1/integrations/payment-status` → PaymentIntegrationController@queryPaymentStatus

### Finance service — `invoice-batch-dispatch-email` endpoints

Area: Configuration

- `GET /v1/invoice-batch-dispatch-email` → InvoiceBatchDispatchEmailController@index
- `POST /v1/invoice-batch-dispatch-email` → InvoiceBatchDispatchEmailController@store
- `GET /v1/invoice-batch-dispatch-email/{id}` → InvoiceBatchDispatchEmailController@show
- `PUT/PATCH /v1/invoice-batch-dispatch-email/{id}` → InvoiceBatchDispatchEmailController@update
- `DELETE /v1/invoice-batch-dispatch-email/{id}` → InvoiceBatchDispatchEmailController@destroy

### Finance service — `invoice-items` endpoints

Area: Cash-specific invoice routes (used by the Create Cash Invoice UI)

- `GET /v1/invoice-items` → InvoiceItemController@index
- `POST /v1/invoice-items` → InvoiceItemController@store
- `GET /v1/invoice-items/{id}` → InvoiceItemController@show
- `PUT/PATCH /v1/invoice-items/{id}` → InvoiceItemController@update
- `DELETE /v1/invoice-items/{id}` → InvoiceItemController@destroy

### Finance service — `invoice-payer` endpoints

Area: Configuration

- `GET /v1/invoice-payer` → InvoicePayerController@index
- `POST /v1/invoice-payer` → InvoicePayerController@store
- `GET /v1/invoice-payer/{id}` → InvoicePayerController@show
- `PUT/PATCH /v1/invoice-payer/{id}` → InvoicePayerController@update
- `DELETE /v1/invoice-payer/{id}` → InvoicePayerController@destroy

### Finance service — `invoice-payment` endpoints

Area: Configuration

- `GET /v1/invoice-payment` → InvoicePaymentController@index
- `POST /v1/invoice-payment` → InvoicePaymentController@store
- `GET /v1/invoice-payment/{id}` → InvoicePaymentController@show
- `PUT/PATCH /v1/invoice-payment/{id}` → InvoicePaymentController@update
- `DELETE /v1/invoice-payment/{id}` → InvoicePaymentController@destroy

### Finance service — `invoices` endpoints

Area: Cash-specific invoice routes (used by the Create Cash Invoice UI); Invoices Management; the value already sent out as the claim's own invoice number.

- `GET /v1/invoices` → InvoiceController@index
- `POST /v1/invoices` → InvoiceController@store
- `GET /v1/invoices/{id}` → InvoiceController@show
- `PUT/PATCH /v1/invoices/{id}` → InvoiceController@update
- `DELETE /v1/invoices/{id}` → InvoiceController@destroy
- `POST /v1/invoices/{invoice}/payments` → InvoiceController@addPayment
- `POST /v1/invoices/sha-payment` → InvoiceController@recordShaPayment (needs permission:insurance_claims.submit|finance.manage)
- `POST /v1/invoices/{invoice}/items` → InvoiceController@addItem
- `POST /v1/invoices/{invoice}/signature` → InvoiceController@updateSignature
- `POST /v1/invoices/{invoice}/exempt` → InvoiceController@exempt (needs permission:waivers.approve|billing.apply_discounts|finance.manage)

### Finance service — `patient-accounts` endpoints

Area: Patient Accounts

- `GET /v1/patient-accounts/balances` → PatientAccountController@balances
- `POST /v1/patient-accounts/deposit` → PatientAccountController@deposit
- `POST /v1/patient-accounts/withdraw` → PatientAccountController@withdraw (needs permission:deposits.refund|finance.manage)
- `GET /v1/patient-accounts/{patientAccount}/transactions` → PatientAccountController@transactions
- `GET /v1/patient-accounts` → PatientAccountController@index
- `POST /v1/patient-accounts` → PatientAccountController@store
- `GET /v1/patient-accounts/{id}` → PatientAccountController@show
- `PUT/PATCH /v1/patient-accounts/{id}` → PatientAccountController@update
- `DELETE /v1/patient-accounts/{id}` → PatientAccountController@destroy

### Finance service — `patient-transaction` endpoints

Area: Configuration

- `GET /v1/patient-transaction` → PatientTransactionController@index
- `POST /v1/patient-transaction` → PatientTransactionController@store
- `GET /v1/patient-transaction/{id}` → PatientTransactionController@show
- `PUT/PATCH /v1/patient-transaction/{id}` → PatientTransactionController@update
- `DELETE /v1/patient-transaction/{id}` → PatientTransactionController@destroy

### Finance service — `patients` endpoints

Area: ===== PATIENT BILLS =====

- `GET /v1/patients/{patientId}/bills` → PatientBillsController@getBills
- `GET /v1/patients/{patientId}/bills/account` → PatientBillsController@getAccountDetails
- `GET /v1/patients/{patientId}/bills/outstanding` → PatientBillsController@getOutstandingBalance
- `POST /v1/patients/{patientId}/bills/invoice/{invoiceId}/items` → PatientBillsController@addItems
- `PUT /v1/patients/{patientId}/bills/item/{itemId}` → PatientBillsController@updateInvoiceItem
- `DELETE /v1/patients/{patientId}/bills/item/{itemId}` → PatientBillsController@removeItem
- `POST /v1/patients/{patientId}/bills/invoice/{invoiceId}/manual-item` → PatientBillsController@createManualItem

### Finance service — `payment-modes` endpoints

Area: Configuration

- `GET /v1/payment-modes` → PaymentModeController@index
- `GET /v1/payment-modes/{id}` → PaymentModeController@show

### Finance service — `payments` endpoints

Area: Payments Management

- `GET /v1/payments` → PaymentController@index
- `POST /v1/payments` → PaymentController@store
- `GET /v1/payments/{id}` → PaymentController@show
- `PUT/PATCH /v1/payments/{id}` → PaymentController@update
- `DELETE /v1/payments/{id}` → PaymentController@destroy
- `POST /v1/payments/{payment}/allocate` → PaymentController@allocate

### Finance service — `payments-card` endpoints

Area: Configuration

- `GET /v1/payments-card` → PaymentsCardController@index
- `POST /v1/payments-card` → PaymentsCardController@store
- `GET /v1/payments-card/{id}` → PaymentsCardController@show
- `PUT/PATCH /v1/payments-card/{id}` → PaymentsCardController@update
- `DELETE /v1/payments-card/{id}` → PaymentsCardController@destroy

### Finance service — `payments-cash` endpoints

Area: Configuration

- `GET /v1/payments-cash` → PaymentsCashController@index
- `POST /v1/payments-cash` → PaymentsCashController@store
- `GET /v1/payments-cash/{id}` → PaymentsCashController@show
- `PUT/PATCH /v1/payments-cash/{id}` → PaymentsCashController@update
- `DELETE /v1/payments-cash/{id}` → PaymentsCashController@destroy

### Finance service — `payments-cheque` endpoints

Area: Configuration

- `GET /v1/payments-cheque` → PaymentsChequeController@index
- `POST /v1/payments-cheque` → PaymentsChequeController@store
- `GET /v1/payments-cheque/{id}` → PaymentsChequeController@show
- `PUT/PATCH /v1/payments-cheque/{id}` → PaymentsChequeController@update
- `DELETE /v1/payments-cheque/{id}` → PaymentsChequeController@destroy

### Finance service — `payments-mpesa` endpoints

Area: Configuration

- `GET /v1/payments-mpesa` → PaymentsMpesaController@index
- `POST /v1/payments-mpesa` → PaymentsMpesaController@store
- `GET /v1/payments-mpesa/{id}` → PaymentsMpesaController@show
- `PUT/PATCH /v1/payments-mpesa/{id}` → PaymentsMpesaController@update
- `DELETE /v1/payments-mpesa/{id}` → PaymentsMpesaController@destroy

### Finance service — `payments-pesa-pal-card` endpoints

Area: Configuration

- `GET /v1/payments-pesa-pal-card` → PaymentsPesaPalCardController@index
- `POST /v1/payments-pesa-pal-card` → PaymentsPesaPalCardController@store
- `GET /v1/payments-pesa-pal-card/{id}` → PaymentsPesaPalCardController@show
- `PUT/PATCH /v1/payments-pesa-pal-card/{id}` → PaymentsPesaPalCardController@update
- `DELETE /v1/payments-pesa-pal-card/{id}` → PaymentsPesaPalCardController@destroy

### Finance service — `payments-pesa-pal-mpesa` endpoints

Area: Configuration

- `GET /v1/payments-pesa-pal-mpesa` → PaymentsPesaPalMpesaController@index
- `POST /v1/payments-pesa-pal-mpesa` → PaymentsPesaPalMpesaController@store
- `GET /v1/payments-pesa-pal-mpesa/{id}` → PaymentsPesaPalMpesaController@show
- `PUT/PATCH /v1/payments-pesa-pal-mpesa/{id}` → PaymentsPesaPalMpesaController@update
- `DELETE /v1/payments-pesa-pal-mpesa/{id}` → PaymentsPesaPalMpesaController@destroy

### Finance service — `paymentsReceipts` endpoints

Area: can hit them directly.

- `POST /v1/paymentsReceipts/{receipt}/cancel` → StandaloneCashPaymentController@cancelReceipt (needs permission:receipts.void|payments.reverse|finance.manage)
- `GET /v1/paymentsReceipts/{receipt}/credit-note-items` → StandaloneCashPaymentController@creditNoteItems (needs permission:credit_notes.create|receipts.void|finance.manage)
- `POST /v1/paymentsReceipts/{receipt}/issue-credit-note` → StandaloneCashPaymentController@issueCreditNote (needs permission:credit_notes.create|receipts.void|finance.manage)
- `POST /v1/paymentsReceipts/{receipt}/approve-credit-note` → StandaloneCashPaymentController@approveCreditNote (needs permission:credit_notes.approve|finance.manage)
- `POST /v1/paymentsReceipts/{receipt}/cancel-credit-note` → StandaloneCashPaymentController@cancelCreditNote (needs permission:credit_notes.void|finance.manage)

### Finance service — `petty-cash-updates` endpoints

Area: Configuration

- `GET /v1/petty-cash-updates` → PettyCashUpdatesController@index
- `POST /v1/petty-cash-updates` → PettyCashUpdatesController@store
- `GET /v1/petty-cash-updates/{id}` → PettyCashUpdatesController@show
- `PUT/PATCH /v1/petty-cash-updates/{id}` → PettyCashUpdatesController@update
- `DELETE /v1/petty-cash-updates/{id}` → PettyCashUpdatesController@destroy

### Finance service — `point` endpoints

Area: Configuration

- `GET /v1/point` → PointController@index
- `POST /v1/point` → PointController@store
- `GET /v1/point/{id}` → PointController@show
- `PUT/PATCH /v1/point/{id}` → PointController@update
- `DELETE /v1/point/{id}` → PointController@destroy

### Finance service — `points` endpoints

Area: Report templates & scheduled reports

- `GET /v1/points/customer` → VoucherPointsController@getCustomerPoints
- `POST /v1/points/redeem` → VoucherPointsController@redeemPoints
- `GET /v1/points/rate` → VoucherPointsController@getPointsRate
- `GET /v1/points/search-by-phone` → VoucherPointsController@searchByPhone

### Finance service — `printing` endpoints

Area: ===== PRINTING =====; needs to precede the apiResource here too, not just down there.

- `GET /v1/printing` → PrintingController@index
- `POST /v1/printing` → PrintingController@store
- `GET /v1/printing/invoice/{invoice}` → PrintingController@printInvoice
- `GET /v1/printing/receipt/{payment}` → PrintingController@printReceipt
- `GET /v1/printing/credit-note/{id}` → PrintingController@printCreditNote
- `GET /v1/printing/statement/{patient}` → PrintingController@printStatement
- `GET /v1/printing/waiver/{id}` → PrintingController@printWaiver
- `POST /v1/printing/bulk/invoices` → PrintingController@bulkInvoices
- `GET /v1/printing/reports/daily-receipts` → PrintingController@dailyReceiptsReport
- `GET /v1/printing/reports/monthly-statement` → PrintingController@monthlyStatement
- `GET /v1/printing/templates` → PrintingController@getTemplates
- `POST /v1/printing/templates` → PrintingController@createTemplate (needs permission:settings.manage_templates|settings.manage|finance.manage)
- `PUT /v1/printing/templates/{id}` → PrintingController@updateTemplate (needs permission:settings.manage_templates|settings.manage|finance.manage)
- `GET /v1/printing/preview/invoice/{invoice}` → PrintingController@previewInvoice
- `GET /v1/printing/insurance-claim/{id}` → PrintingController@printInsuranceClaim
- `GET /v1/printing/voucher/{id}` → PrintingController@printVoucher
- `POST /v1/printing/bulk/documents` → PrintingController@bulkDocuments
- `GET /v1/printing/jobs/{jobId}` → PrintingController@getJobStatus

### Finance service — `receipt` endpoints

Area: ===== PRINTING =====

- `POST /v1/receipt/{receipt}/credit-note` → StandaloneCashPaymentController@issueCreditNote (needs permission:credit_notes.create|receipts.void|finance.manage)

### Finance service — `reports` endpoints

Area: ===== REPORTS =====; Report templates & scheduled reports; needs to precede the apiResource here too, not just down there.

- `GET /v1/reports` → ReportController@index
- `POST /v1/reports` → ReportController@store
- `GET /v1/reports/billing` → ReportController@billingReport
- `GET /v1/reports/billing/export` → ReportController@billingExport
- `GET /v1/reports/payment` → ReportController@paymentReport
- `GET /v1/reports/payments` → ReportController@paymentReport
- `GET /v1/reports/aging-analysis` → ReportController@agingAnalysis
- `GET /v1/reports/aging` → ReportController@agingAnalysis
- `GET /v1/reports/reconciliation` → ReportController@reconciliation
- `GET /v1/reports/revenue` → ReportController@revenueReport
- `GET /v1/reports/patient-statement/{patientId}` → ReportController@patientStatement
- `GET /v1/reports/insurance-claims` → ReportController@insuranceClaims
- `GET /v1/reports/credit-notes` → ReportController@creditNotes
- `GET /v1/reports/waivers` → ReportController@waivers
- `GET /v1/reports/dashboard-summary` → ReportController@dashboardSummary
- `GET /v1/reports/download/{filename}` → ReportController@downloadReport
- `GET /v1/reports/templates` → ReportController@getTemplates
- `POST /v1/reports/templates` → ReportController@createTemplate
- `GET /v1/reports/scheduled` → ReportController@getScheduled
- `POST /v1/reports/scheduled` → ReportController@createScheduled
- `GET /v1/reports/patient/{patientId}/voucher/{voucherId}/print` → VoucherPointsController@printVoucher

### Finance service — `revenue-dump` endpoints

Area: Configuration

- `GET /v1/revenue-dump` → RevenueDumpController@index
- `POST /v1/revenue-dump` → RevenueDumpController@store
- `GET /v1/revenue-dump/{id}` → RevenueDumpController@show
- `PUT/PATCH /v1/revenue-dump/{id}` → RevenueDumpController@update
- `DELETE /v1/revenue-dump/{id}` → RevenueDumpController@destroy

### Finance service — `service-charges` endpoints

Area: Configuration

- `GET /v1/service-charges` → ServiceChargeController@index
- `POST /v1/service-charges` → ServiceChargeController@store
- `GET /v1/service-charges/{id}` → ServiceChargeController@show
- `PUT/PATCH /v1/service-charges/{id}` → ServiceChargeController@update
- `DELETE /v1/service-charges/{id}` → ServiceChargeController@destroy

### Finance service — `sha` endpoints

Area: SHA claim builds — the Claim builds tab (ShaClaimBuilds).

- `GET /v1/sha/claim-builds` → ShaClaimBuildController@index
- `POST /v1/sha/claim-builds/process-due` → ShaClaimBuildController@processDue
- `POST /v1/sha/claim-builds/{build}/retry` → ShaClaimBuildController@retry

### Finance service — `split-bills` endpoints

Area: Read-only index/show onto split_bills.

- `GET /v1/split-bills` → SplitBillsController@index
- `GET /v1/split-bills/{id}` → SplitBillsController@show

### Finance service — `supplier-invoices` endpoints

Area: Supplier invoices (accounts payable, from procurement GRN/direct-order receipt)

- `GET /v1/supplier-invoices` → SupplierInvoiceController@index
- `POST /v1/supplier-invoices` → SupplierInvoiceController@store
- `GET /v1/supplier-invoices/{id}` → SupplierInvoiceController@show
- `POST /v1/supplier-invoices/{id}/payments` → SupplierInvoiceController@addPayment (needs permission:payments.process|finance.manage)

### Finance service — `theatre-finance` endpoints

Area: supplies raw operation/consumable data over the gateway.

- `GET /v1/theatre-finance/billable` → TheatreFinanceController@billable

### Finance service — `theatre-operations` endpoints

Area: supplies raw operation/consumable data over the gateway.

- `GET /v1/theatre-operations/{operation}/charge-sheet` → TheatreFinanceController@chargeSheet
- `POST /v1/theatre-operations/{operation}/mark-billed` → TheatreFinanceController@markBilled

### Finance service — `transactions-v2` endpoints

Area: Configuration

- `GET /v1/transactions-v2` → TransactionsController@index
- `POST /v1/transactions-v2` → TransactionsController@store
- `GET /v1/transactions-v2/{id}` → TransactionsController@show
- `PUT/PATCH /v1/transactions-v2/{id}` → TransactionsController@update
- `DELETE /v1/transactions-v2/{id}` → TransactionsController@destroy

### Finance service — `voucher` endpoints

Area: Configuration

- `GET /v1/voucher` → VoucherController@index
- `POST /v1/voucher` → VoucherController@store
- `GET /v1/voucher/{id}` → VoucherController@show
- `PUT/PATCH /v1/voucher/{id}` → VoucherController@update
- `DELETE /v1/voucher/{id}` → VoucherController@destroy

### Finance service — `waivers` endpoints

Area: ===== WAIVERS =====; needs to precede the apiResource here too, not just down there.

- `GET /v1/waivers` → WaiverController@index
- `POST /v1/waivers` → WaiverController@store
- `GET /v1/waivers/{id}` → WaiverController@show
- `PUT/PATCH /v1/waivers/{id}` → WaiverController@update
- `DELETE /v1/waivers/{id}` → WaiverController@destroy
- `GET /v1/waivers/invoice/{invoice}` → WaiverController@forInvoice
- `GET /v1/waivers` → WaiverController@index
- `POST /v1/waivers` → WaiverController@store
- `GET /v1/waivers/{id}` → WaiverController@show
- `PUT/PATCH /v1/waivers/{id}` → WaiverController@update
- `DELETE /v1/waivers/{id}` → WaiverController@destroy
- `POST /v1/waivers/{waiver}/approve` → WaiverController@approve (needs permission:waivers.approve|finance.manage)
- `POST /v1/waivers/{waiver}/reject` → WaiverController@reject (needs permission:waivers.reject|finance.manage)
- `POST /v1/waivers/{waiver}/apply` → WaiverController@apply (needs permission:waivers.approve|finance.manage)

## Inpatient service API (inpatient-service)

Covers: wards, beds, admissions, charge sheets, discharges. 260 endpoints.

### Inpatient service — `admission-requests` endpoints

Area: Admission Requests; changed in between.; the request without also being able to approve or admit.

- `GET /v1/admission-requests` → AdmissionRequestController@index
- `POST /v1/admission-requests` → AdmissionRequestController@store (needs permission:inpatient.admit_patients|inpatient.manage)
- `POST /v1/admission-requests/from-visit` → AdmissionRequestController@fromVisit (needs permission:inpatient.admission_requests.create_from_reception|inpatient.admit_patients|inpatient.manage)
- `GET /v1/admission-requests/pending` → AdmissionRequestController@pending
- `GET /v1/admission-requests/quote` → AdmissionRequestController@quote
- `GET /v1/admission-requests/approved` → AdmissionRequestController@approved
- `GET /v1/admission-requests/{id}` → AdmissionRequestController@show
- `PUT /v1/admission-requests/{id}` → AdmissionRequestController@update (needs permission:inpatient.admit_patients|inpatient.manage)
- `DELETE /v1/admission-requests/{id}` → AdmissionRequestController@destroy (needs permission:inpatient.admit_patients|inpatient.manage)
- `POST /v1/admission-requests/{id}/approve` → AdmissionRequestController@approve (needs permission:inpatient.admit_patients|inpatient.manage)
- `POST /v1/admission-requests/{id}/reject` → AdmissionRequestController@reject (needs permission:inpatient.admit_patients|inpatient.manage)
- `POST /v1/admission-requests/{id}/cancel` → AdmissionRequestController@cancel (needs permission:inpatient.admit_patients|inpatient.manage)
- `POST /v1/admission-requests/{id}/authorize` → AdmissionRequestController@authorize (needs permission:inpatient.admit_patients|inpatient.manage)
- `POST /v1/admission-requests/{id}/create-admission` → AdmissionRequestController@createAdmission (needs permission:inpatient.admit_patients|inpatient.manage)
- `GET /v1/admission-requests/{id}/deposit-quote` → AdmissionRequestController@depositQuote
- `GET /v1/admission-requests/{id}/logs` → AdmissionRequestController@logs
- `GET /v1/admission-requests/{id}/print` → AdmissionRequestController@print

### Inpatient service — `admission-types` endpoints

Area: Admission Types

- `GET /v1/admission-types` → AdmissionTypeController@index
- `POST /v1/admission-types` → AdmissionTypeController@store
- `GET /v1/admission-types/{id}` → AdmissionTypeController@show
- `PUT/PATCH /v1/admission-types/{id}` → AdmissionTypeController@update
- `DELETE /v1/admission-types/{id}` → AdmissionTypeController@destroy

### Inpatient service — `admissions` endpoints

Area: Admissions; Bill Breakdowns; Visit Rebates; a facility can rename - see the controller.

- `GET /v1/admissions` → AdmissionController@index
- `POST /v1/admissions` → AdmissionController@store (needs permission:inpatient.admit_patients|inpatient.manage)
- `GET /v1/admissions/active` → AdmissionController@active
- `GET /v1/admissions/awaiting-discharge` → AdmissionController@awaitingDischarge
- `GET /v1/admissions/queue` → AdmissionController@queue
- `GET /v1/admissions/summary` → AdmissionController@summary
- `GET /v1/admissions/generate-inpatient-number` → AdmissionController@generateInpatientNumber
- `GET /v1/admissions/{id}` → AdmissionController@show
- `PUT /v1/admissions/{id}` → AdmissionController@update (needs permission:inpatient.admit_patients|inpatient.manage)
- `POST /v1/admissions/{id}/edit-admission-details` → AdmissionController@editAdmissionDetails (needs permission:inpatient.admit_patients|inpatient.manage)
- `POST /v1/admissions/{id}/send-to-morgue` → AdmissionController@sendToMorgue (needs permission:morgue.intake.request|inpatient.discharge_patients|inpatient.manage)
- `GET /v1/admissions/{id}/charge-sheet` → AdmissionController@chargeSheet
- `GET /v1/admissions/{id}/daily-charges` → ChargeSheetController@dailyCharges
- `POST /v1/admissions/{id}/daily-charges` → ChargeSheetController@addCharge
- `GET /v1/admissions/{id}/print/daily-chargesheet` → ChargeSheetController@printDailyChargesheet
- `GET /v1/admissions/{id}/print/daily-invoice` → ChargeSheetController@printDailyInvoice
- `POST /v1/admissions/{id}/verify-day-bills` → ChargeSheetController@verifyDayBills
- `GET /v1/admissions/{id}/verifications` → ChargeSheetController@verifications
- `POST /v1/admissions/{id}/finalize-chargesheet` → ChargeSheetController@finalizeChargesheet
- `POST /v1/admissions/{id}/update-deposit` → AdmissionController@updateDeposit (needs permission:inpatient.admit_patients|inpatient.manage|deposits.create|finance.manage)
- `GET /v1/admissions/{id}/rebates/available` → VisitRebateController@available
- `GET /v1/admissions/{id}/rebates` → VisitRebateController@index
- `POST /v1/admissions/{id}/rebates` → VisitRebateController@store (needs permission:billing.apply_discounts|waivers.approve|finance.manage)
- `DELETE /v1/admissions/{id}/rebates/{rebateId}` → VisitRebateController@destroy (needs permission:billing.apply_discounts|waivers.approve|finance.manage)
- `GET /v1/admissions/{id}/bill-breakdowns` → BillBreakdownController@index
- `POST /v1/admissions/{id}/bill-breakdowns` → BillBreakdownController@store (needs permission:inpatient.admit_patients|billing.manage|finance.manage)
- `DELETE /v1/admissions/{id}/bill-breakdowns/{breakdownId}` → BillBreakdownController@destroy (needs permission:inpatient.admit_patients|billing.manage|finance.manage)

### Inpatient service — `bed-types` endpoints

Area: Bed Types

- `GET /v1/bed-types` → BedTypeController@index
- `POST /v1/bed-types` → BedTypeController@store
- `GET /v1/bed-types/{id}` → BedTypeController@show
- `PUT/PATCH /v1/bed-types/{id}` → BedTypeController@update
- `DELETE /v1/bed-types/{id}` → BedTypeController@destroy

### Inpatient service — `beds` endpoints

Area: Beds

- `GET /v1/beds` → BedController@index
- `POST /v1/beds` → BedController@store (needs permission:beds.create|inpatient.manage_wards)
- `GET /v1/beds/available` → BedController@available
- `GET /v1/beds/by-ward/{wardId}` → BedController@byWard
- `GET /v1/beds/{id}` → BedController@show
- `PUT /v1/beds/{id}` → BedController@update (needs permission:beds.update|inpatient.manage_wards)
- `DELETE /v1/beds/{id}` → BedController@destroy (needs permission:beds.delete|inpatient.manage_wards)
- `POST /v1/beds/{id}/allocate` → BedController@allocate
- `POST /v1/beds/{id}/release` → BedController@release
- `POST /v1/beds/{id}/reserve` → BedController@reserve
- `POST /v1/beds/{id}/maintenance` → BedController@maintenance

### Inpatient service — `blood-transfusions` endpoints

Area: Blood Transfusions

- `GET /v1/blood-transfusions` → BloodTransfusionController@index
- `POST /v1/blood-transfusions` → BloodTransfusionController@store
- `GET /v1/blood-transfusions/{id}` → BloodTransfusionController@show
- `PUT/PATCH /v1/blood-transfusions/{id}` → BloodTransfusionController@update
- `DELETE /v1/blood-transfusions/{id}` → BloodTransfusionController@destroy

### Inpatient service — `care-plans` endpoints

Area: Additional v2 parity models

- `GET /v1/care-plans` → CarePlanController@index
- `POST /v1/care-plans` → CarePlanController@store
- `GET /v1/care-plans/{id}` → CarePlanController@show
- `PUT/PATCH /v1/care-plans/{id}` → CarePlanController@update
- `DELETE /v1/care-plans/{id}` → CarePlanController@destroy
- `POST /v1/care-plans/{id}/review` → CarePlanController@review

### Inpatient service — `charge-sheet` endpoints

Area: Charge Sheet

- `GET /v1/charge-sheet/by-admission/{admissionId}` → ChargeSheetController@byAdmission
- `POST /v1/charge-sheet/add-charge` → ChargeSheetController@addCharge
- `PUT /v1/charge-sheet/update-charge/{chargeId}` → ChargeSheetController@updateCharge
- `POST /v1/charge-sheet/cancel-charge/{chargeId}` → ChargeSheetController@cancelCharge
- `POST /v1/charge-sheet/approve/{admissionId}` → ChargeSheetController@approve
- `GET /v1/charge-sheet/change-history/{admissionId}` → ChargeSheetController@changeHistory

### Inpatient service — `charge-sheets` endpoints

Area: (Dynamics item number) — read by finance when it sends an invoice.; billing-sync uses — a cashier's token carries no inpatient permission.

- `POST /v1/charge-sheets/bill` → ChargeSheetController@bill (needs permission:finance.process_payments|billing.manage|finance.manage|inpatient.manage)
- `GET /v1/charge-sheets/procedures` → ChargeSheetController@procedures

### Inpatient service — `charges` endpoints

Area: Charges

- `GET /v1/charges` → ChargeController@index
- `POST /v1/charges` → ChargeController@store
- `GET /v1/charges/{id}` → ChargeController@show
- `PUT/PATCH /v1/charges/{id}` → ChargeController@update
- `DELETE /v1/charges/{id}` → ChargeController@destroy

### Inpatient service — `discharge-notes` endpoints

Area: Additional v2 parity models

- `GET /v1/discharge-notes` → DischargeNoteController@index
- `POST /v1/discharge-notes` → DischargeNoteController@store
- `GET /v1/discharge-notes/{id}` → DischargeNoteController@show
- `PUT/PATCH /v1/discharge-notes/{id}` → DischargeNoteController@update
- `DELETE /v1/discharge-notes/{id}` → DischargeNoteController@destroy

### Inpatient service — `discharge-requests` endpoints

Area: Discharge Requests

- `GET /v1/discharge-requests` → DischargeRequestController@index
- `POST /v1/discharge-requests` → DischargeRequestController@store (needs permission:discharge.initiate|inpatient.discharge_patients)
- `GET /v1/discharge-requests/pending` → DischargeRequestController@pending
- `GET /v1/discharge-requests/{id}` → DischargeRequestController@show
- `PUT /v1/discharge-requests/{id}` → DischargeRequestController@update (needs permission:discharge.initiate|inpatient.discharge_patients)
- `DELETE /v1/discharge-requests/{id}` → DischargeRequestController@destroy (needs permission:discharge.cancel|inpatient.discharge_patients)
- `POST /v1/discharge-requests/{id}/approve` → DischargeRequestController@approve (needs permission:discharge.approve|inpatient.discharge_patients)

### Inpatient service — `discharge-types` endpoints

Area: Discharge Types

- `GET /v1/discharge-types` → DischargeTypeController@index
- `POST /v1/discharge-types` → DischargeTypeController@store
- `GET /v1/discharge-types/{id}` → DischargeTypeController@show
- `PUT/PATCH /v1/discharge-types/{id}` → DischargeTypeController@update
- `DELETE /v1/discharge-types/{id}` → DischargeTypeController@destroy

### Inpatient service — `discharges` endpoints

Area: Discharges

- `GET /v1/discharges` → DischargeController@index
- `GET /v1/discharges/history` → DischargeController@history
- `GET /v1/discharges/{id}` → DischargeController@show
- `POST /v1/discharges/discharge` → DischargeController@discharge (needs permission:discharge.complete|inpatient.discharge_patients)
- `POST /v1/discharges/{admissionId}/clear-bills` → DischargeController@clearBills

### Inpatient service — `drug-administration` endpoints

Area: Drug Administration

- `GET /v1/drug-administration` → DrugAdministrationController@index
- `POST /v1/drug-administration` → DrugAdministrationController@store
- `GET /v1/drug-administration/by-admission/{admissionId}` → DrugAdministrationController@byAdmission
- `GET /v1/drug-administration/due-now` → DrugAdministrationController@dueNow
- `GET /v1/drug-administration/missed` → DrugAdministrationController@missed
- `GET /v1/drug-administration/{id}` → DrugAdministrationController@show
- `PUT /v1/drug-administration/{id}` → DrugAdministrationController@update
- `DELETE /v1/drug-administration/{id}` → DrugAdministrationController@destroy
- `POST /v1/drug-administration/{id}/administer` → DrugAdministrationController@administer
- `POST /v1/drug-administration/{id}/mark-missed` → DrugAdministrationController@markMissed

### Inpatient service — `feeding-plans` endpoints

Area: Additional v2 parity models

- `GET /v1/feeding-plans` → FeedingPlanController@index
- `POST /v1/feeding-plans` → FeedingPlanController@store
- `GET /v1/feeding-plans/{id}` → FeedingPlanController@show
- `PUT/PATCH /v1/feeding-plans/{id}` → FeedingPlanController@update
- `DELETE /v1/feeding-plans/{id}` → FeedingPlanController@destroy

### Inpatient service — `fluid-balances` endpoints

Area: Additional v2 parity models

- `GET /v1/fluid-balances` → FluidBalanceController@index
- `POST /v1/fluid-balances` → FluidBalanceController@store
- `GET /v1/fluid-balances/{id}` → FluidBalanceController@show
- `PUT/PATCH /v1/fluid-balances/{id}` → FluidBalanceController@update
- `DELETE /v1/fluid-balances/{id}` → FluidBalanceController@destroy

### Inpatient service — `gateway` endpoints

Area: Model Gateway — cross-tenant data introspection and setup copying

- `POST /v1/gateway` → ModelGatewayController@handle

### Inpatient service — `health` endpoints

Area: Health check endpoint (public, unversioned)

- `GET /health` → HealthCheckController@check

### Inpatient service — `inpatient-history` endpoints

Area: Additional v2 parity models

- `GET /v1/inpatient-history/{visitId}` → InpatientHistoryController@show

### Inpatient service — `labour-board` endpoints

Area: facilities piloting the structured LCG. See build spec §6.

- `GET /v1/labour-board` → LabourBoardController@index

### Inpatient service — `labour-episodes` endpoints

Area: facilities piloting the structured LCG. See build spec §6.

- `GET /v1/labour-episodes` → LabourEpisodeController@index
- `POST /v1/labour-episodes` → LabourEpisodeController@store (needs permission:inpatient.manage)
- `GET /v1/labour-episodes/by-admission/{admissionId}` → LabourEpisodeController@byAdmission
- `GET /v1/labour-episodes/{uuid}/grid` → LabourEpisodeController@grid
- `POST /v1/labour-episodes/{uuid}/second-stage` → LabourEpisodeController@secondStage (needs permission:inpatient.manage)
- `POST /v1/labour-episodes/{uuid}/observations` → LcgObservationController@store (needs permission:inpatient.manage)
- `POST /v1/labour-episodes/{uuid}/observations/{id}/correct` → LcgObservationController@correct (needs permission:inpatient.manage)
- `POST /v1/labour-episodes/{uuid}/notes` → LcgNoteController@store (needs permission:inpatient.manage)
- `GET /v1/labour-episodes/{uuid}/summary` → LabourSummaryController@show
- `POST /v1/labour-episodes/{uuid}/summary` → LabourSummaryController@store (needs permission:inpatient.manage)
- `GET /v1/labour-episodes/{uuid}/sheet.pdf` → LcgSheetController@show
- `PATCH /v1/labour-episodes/{uuid}/newborns/{id}/link-patient` → NewbornRecordController@linkPatient (needs permission:inpatient.manage)

### Inpatient service — `lcg-alerts` endpoints

Area: facilities piloting the structured LCG. See build spec §6.

- `POST /v1/lcg-alerts/{id}/acknowledge` → LcgAlertController@acknowledge (needs permission:inpatient.manage)
- `POST /v1/lcg-alerts/{id}/escalate` → LcgAlertController@escalate (needs permission:inpatient.manage)
- `POST /v1/lcg-alerts/{id}/resolve` → LcgAlertController@resolve (needs permission:inpatient.manage)

### Inpatient service — `logs` endpoints

Area: Admission Logs

- `GET /v1/logs/by-admission/{admissionId}` → AdmissionLogController@byAdmission

### Inpatient service — `mccod` endpoints

Area: Discharge Requests

- `GET /v1/mccod` → MccodController@show (needs permission:discharge.initiate|inpatient.discharge_patients)
- `POST /v1/mccod` → MccodController@store (needs permission:discharge.initiate|inpatient.discharge_patients)

### Inpatient service — `notes` endpoints

Area: Notes

- `GET /v1/notes` → NoteController@index
- `POST /v1/notes` → NoteController@store
- `GET /v1/notes/by-admission/{admissionId}` → NoteController@byAdmission
- `GET /v1/notes/by-visit/{visitId}` → NoteController@byVisit
- `GET /v1/notes/by-type/{type}` → NoteController@byType
- `GET /v1/notes/view-logs` → NoteController@viewLogs
- `POST /v1/notes/log-view` → NoteController@logView
- `GET /v1/notes/{id}` → NoteController@show
- `PUT /v1/notes/{id}` → NoteController@update
- `DELETE /v1/notes/{id}` → NoteController@destroy

### Inpatient service — `nursing-care-plans` endpoints

Area: Nursing Care Plans

- `GET /v1/nursing-care-plans` → NursingCarePlanController@index
- `POST /v1/nursing-care-plans` → NursingCarePlanController@store
- `GET /v1/nursing-care-plans/by-admission/{admissionId}` → NursingCarePlanController@byAdmission
- `GET /v1/nursing-care-plans/{id}` → NursingCarePlanController@show
- `PUT /v1/nursing-care-plans/{id}` → NursingCarePlanController@update
- `DELETE /v1/nursing-care-plans/{id}` → NursingCarePlanController@destroy
- `POST /v1/nursing-care-plans/{id}/review` → NursingCarePlanController@review
- `POST /v1/nursing-care-plans/{id}/discontinue` → NursingCarePlanController@discontinue

### Inpatient service — `obstetric-histories` endpoints

Area: Additional v2 parity models

- `GET /v1/obstetric-histories` → ObstetricHistoryController@index
- `POST /v1/obstetric-histories` → ObstetricHistoryController@store
- `GET /v1/obstetric-histories/{id}` → ObstetricHistoryController@show
- `PUT/PATCH /v1/obstetric-histories/{id}` → ObstetricHistoryController@update
- `DELETE /v1/obstetric-histories/{id}` → ObstetricHistoryController@destroy

### Inpatient service — `partograph-vitals` endpoints

Area: Additional v2 parity models

- `GET /v1/partograph-vitals` → PartographVitalsController@index
- `POST /v1/partograph-vitals` → PartographVitalsController@store
- `GET /v1/partograph-vitals/{id}` → PartographVitalsController@show
- `PUT/PATCH /v1/partograph-vitals/{id}` → PartographVitalsController@update
- `DELETE /v1/partograph-vitals/{id}` → PartographVitalsController@destroy

### Inpatient service — `partographs` endpoints

Area: Additional v2 parity models

- `GET /v1/partographs` → PartographController@index
- `POST /v1/partographs` → PartographController@store
- `GET /v1/partographs/{id}` → PartographController@show
- `PUT/PATCH /v1/partographs/{id}` → PartographController@update
- `DELETE /v1/partographs/{id}` → PartographController@destroy

### Inpatient service — `patient-management` endpoints

Area: Additional v2 parity models

- `GET /v1/patient-management` → PatientManagementController@index
- `POST /v1/patient-management` → PatientManagementController@store
- `GET /v1/patient-management/{id}` → PatientManagementController@show
- `PUT/PATCH /v1/patient-management/{id}` → PatientManagementController@update
- `DELETE /v1/patient-management/{id}` → PatientManagementController@destroy

### Inpatient service — `post-natal` endpoints

Area: Additional v2 parity models

- `GET /v1/post-natal` → PostNatalController@index
- `POST /v1/post-natal` → PostNatalController@store
- `GET /v1/post-natal/{id}` → PostNatalController@show
- `PUT/PATCH /v1/post-natal/{id}` → PostNatalController@update
- `DELETE /v1/post-natal/{id}` → PostNatalController@destroy

### Inpatient service — `preliminaries` endpoints

Area: Additional v2 parity models

- `GET /v1/preliminaries` → PreliminaryController@index
- `POST /v1/preliminaries` → PreliminaryController@store
- `GET /v1/preliminaries/{id}` → PreliminaryController@show
- `PUT/PATCH /v1/preliminaries/{id}` → PreliminaryController@update
- `DELETE /v1/preliminaries/{id}` → PreliminaryController@destroy

### Inpatient service — `preoperatives` endpoints

Area: Additional v2 parity models

- `GET /v1/preoperatives` → PreoperativeController@index
- `POST /v1/preoperatives` → PreoperativeController@store
- `GET /v1/preoperatives/{id}` → PreoperativeController@show
- `PUT/PATCH /v1/preoperatives/{id}` → PreoperativeController@update
- `DELETE /v1/preoperatives/{id}` → PreoperativeController@destroy

### Inpatient service — `procedures` endpoints

Area: Procedures

- `GET /v1/procedures` → ProcedureController@index
- `POST /v1/procedures` → ProcedureController@store
- `GET /v1/procedures/pending-approval` → ProcedureController@pendingApproval
- `GET /v1/procedures/{id}` → ProcedureController@show
- `PUT /v1/procedures/{id}` → ProcedureController@update
- `DELETE /v1/procedures/{id}` → ProcedureController@destroy
- `POST /v1/procedures/{id}/approve` → ProcedureController@approve
- `POST /v1/procedures/{id}/reject` → ProcedureController@reject
- `POST /v1/procedures/{id}/complete` → ProcedureController@complete

### Inpatient service — `reception` endpoints

Area: POST admission-request open with no authentication at all.

- `GET /v1/reception/can-check-in` → ReceptionController@canCheckInToInpatient
- `POST /v1/reception/admission-request` → ReceptionController@createAdmissionRequest
- `GET /v1/reception/active-admission` → ReceptionController@getActiveAdmission

### Inpatient service — `reports` endpoints

Area: Reports

- `GET /v1/reports/admission-statistics` → ReportController@admissionStatistics
- `GET /v1/reports/bed-occupancy` → ReportController@bedOccupancy
- `GET /v1/reports/revenue-summary` → ReportController@revenueSummary
- `GET /v1/reports/average-length-of-stay` → ReportController@averageLengthOfStay

### Inpatient service — `temperatures` endpoints

Area: Temperatures

- `GET /v1/temperatures` → TemperatureController@index
- `POST /v1/temperatures` → TemperatureController@store
- `GET /v1/temperatures/{id}` → TemperatureController@show
- `PUT/PATCH /v1/temperatures/{id}` → TemperatureController@update
- `DELETE /v1/temperatures/{id}` → TemperatureController@destroy

### Inpatient service — `transfusions` endpoints

Area: Blood Transfusions

- `GET /v1/transfusions/by-admission/{admissionId}` → BloodTransfusionController@byAdmission
- `GET /v1/transfusions` → BloodTransfusionController@index
- `POST /v1/transfusions` → BloodTransfusionController@store
- `GET /v1/transfusions/{id}` → BloodTransfusionController@show
- `PUT /v1/transfusions/{id}` → BloodTransfusionController@update
- `DELETE /v1/transfusions/{id}` → BloodTransfusionController@destroy

### Inpatient service — `visits` endpoints

Area: billing-sync uses — a cashier's token carries no inpatient permission.

- `GET /v1/visits/{visitId}/billable-charges` → ChargeSheetController@billableForVisit

### Inpatient service — `vitals` endpoints

Area: Vitals

- `GET /v1/vitals` → VitalController@index
- `POST /v1/vitals` → VitalController@store
- `GET /v1/vitals/by-admission/{admissionId}` → VitalController@byAdmission
- `GET /v1/vitals/chart/{admissionId}` → VitalController@chart
- `GET /v1/vitals/{id}` → VitalController@show
- `PUT /v1/vitals/{id}` → VitalController@update
- `DELETE /v1/vitals/{id}` → VitalController@destroy

### Inpatient service — `ward-charges` endpoints

Area: Ward Charges

- `GET /v1/ward-charges` → WardChargeController@index
- `POST /v1/ward-charges` → WardChargeController@store
- `GET /v1/ward-charges/{id}` → WardChargeController@show
- `PUT/PATCH /v1/ward-charges/{id}` → WardChargeController@update
- `DELETE /v1/ward-charges/{id}` → WardChargeController@destroy

### Inpatient service — `ward-transfers` endpoints

Area: Ward Transfers

- `POST /v1/ward-transfers/transfer` → WardTransferController@transfer
- `GET /v1/ward-transfers/history/{admissionId}` → WardTransferController@history

### Inpatient service — `wards` endpoints

Area: Wards

- `GET /v1/wards` → WardController@index
- `POST /v1/wards` → WardController@store (needs permission:wards.create|inpatient.manage_wards)
- `GET /v1/wards/occupancy` → WardController@occupancy
- `GET /v1/wards/types` → WardController@types
- `GET /v1/wards/actions` → WardController@actionsCatalog
- `GET /v1/wards/{id}` → WardController@show
- `PUT /v1/wards/{id}` → WardController@update (needs permission:wards.update|inpatient.manage_wards)
- `DELETE /v1/wards/{id}` → WardController@destroy (needs permission:wards.delete|inpatient.manage_wards)
- `GET /v1/wards/{id}/patients` → WardController@patients
- `GET /v1/wards/{id}/available-beds` → WardController@availableBeds
- `GET /v1/wards/{id}/summary` → WardController@summary
- `GET /v1/wards/{id}/actions` → WardController@actionsForWard

## Integration service API (integration-service)

Covers: per-organization integration switches. 38 endpoints.

### Integration service — `integration-credentials` endpoints

Area: operation checks its own permissions in the controller.

- `GET /v1/integration-credentials/{uuid}`
- `PUT /v1/integration-credentials/{uuid}`
- `DELETE /v1/integration-credentials/{uuid}`
- `POST /v1/integration-credentials/{uuid}/activate`
- `POST /v1/integration-credentials/{uuid}/deactivate`
- `POST /v1/integration-credentials/{uuid}/rotate`
- `POST /v1/integration-credentials/{uuid}/revoke`

### Integration service — `integration-endpoints` endpoints

Area: operation checks its own permissions in the controller.

- `GET /v1/integration-endpoints/{uuid}` → IntegrationEndpointController@show
- `PUT /v1/integration-endpoints/{uuid}` → IntegrationEndpointController@update
- `DELETE /v1/integration-endpoints/{uuid}` → IntegrationEndpointController@destroy
- `POST /v1/integration-endpoints/{uuid}/activate` → IntegrationEndpointController@activate
- `POST /v1/integration-endpoints/{uuid}/deactivate` → IntegrationEndpointController@deactivate
- `POST /v1/integration-endpoints/{uuid}/testable` → IntegrationEndpointController@setTestable
- `POST /v1/integration-endpoints/{uuid}/test` → IntegrationEndpointController@test

### Integration service — `integration-environments` endpoints

Area: operation checks its own permissions in the controller.

- `GET /v1/integration-environments/{environmentUuid}/endpoints` → IntegrationEndpointController@index
- `POST /v1/integration-environments/{environmentUuid}/endpoints` → IntegrationEndpointController@store
- `GET /v1/integration-environments/{environmentUuid}/credentials`
- `POST /v1/integration-environments/{environmentUuid}/credentials`

### Integration service — `integrations` endpoints

Area: operation checks its own permissions in the controller.

- `GET /v1/integrations`
- `POST /v1/integrations`
- `GET /v1/integrations/{uuid}`
- `PUT /v1/integrations/{uuid}`
- `DELETE /v1/integrations/{uuid}`
- `POST /v1/integrations/{uuid}/activate`
- `POST /v1/integrations/{uuid}/deactivate`

### Integration service — `service` endpoints

Area: Which of a batch of correlation ids the provider accepted (accounting:reconcile).; tenant-integrations page reads and writes these directly.

- `PUT /v1/service/organizations/{organizationId}/integrations/{code}` → ServiceTenantIntegrationController@upsert
- `DELETE /v1/service/organizations/{organizationId}/integrations/{code}` → ServiceTenantIntegrationController@deactivate
- `GET /v1/service/organizations/{organizationId}/integrations` → ServiceTenantEnvironmentController@index
- `PUT /v1/service/organizations/{organizationId}/integrations/{code}/environments/{environment}` → ServiceTenantEnvironmentController@upsert
- `DELETE /v1/service/organizations/{organizationId}/integrations/{code}/environments/{environment}` → ServiceTenantEnvironmentController@destroy
- `POST /v1/service/organizations/{organizationId}/integrations/{code}/operations/{operation}` → ServiceIntegrationOperationController@execute
- `POST /v1/service/organizations/{organizationId}/transactions/confirmed` → ServiceIntegrationTransactionController@confirmed
- `GET /v1/service/organizations/{organizationId}/transactions/{correlationId}` → ServiceIntegrationTransactionController@index

### Integration service — `tenant` endpoints

Area: Which of a batch of correlation ids the provider accepted (accounting:reconcile).; operation checks its own permissions in the controller.

- `GET /v1/tenant/integrations` → TenantIntegrationController@index (needs permission:settings.view|settings.edit|settings.manage)
- `PATCH /v1/tenant/integrations/{uuid}` → TenantIntegrationController@update (needs permission:settings.edit|settings.manage)
- `PATCH /v1/tenant/integration-endpoints/{uuid}` → TenantIntegrationController@updateEndpoint (needs permission:settings.edit|settings.manage)
- `GET /v1/tenant/integrations/{code}/operations` → TenantIntegrationOperationController@index
- `POST /v1/tenant/integrations/{code}/operations/{operation}` → TenantIntegrationOperationController@execute

## Inventory service API (inventory-service)

Covers: products, stores, suppliers, requisitions, purchase orders, GRNs, POS. 243 endpoints.

### Inventory service — `alerts` endpoints

Area: Alerts

- `GET /v1/alerts/low-stock` → AlertController@lowStock
- `GET /v1/alerts/expiring-items` → AlertController@expiringItems
- `GET /v1/alerts/summary` → AlertController@summary
- `GET /v1/alerts/critical` → AlertController@critical

### Inventory service — `approval-levels` endpoints

Area: Approval Levels (requisition workflow configuration)

- `GET /v1/approval-levels` → ApprovalLevelController@index
- `POST /v1/approval-levels` → ApprovalLevelController@store
- `DELETE /v1/approval-levels/{id}` → ApprovalLevelController@destroy

### Inventory service — `batches` endpoints

Area: Batches

- `GET /v1/batches/{id}/info` → BatchController@batchInfo
- `GET /v1/batches/expiring` → BatchController@expiring
- `GET /v1/batches/expired` → BatchController@expired
- `GET /v1/batches/lookup` → BatchController@lookup
- `GET /v1/batches` → BatchController@index
- `POST /v1/batches` → BatchController@store
- `GET /v1/batches/{id}` → BatchController@show
- `PUT/PATCH /v1/batches/{id}` → BatchController@update
- `DELETE /v1/batches/{id}` → BatchController@destroy

### Inventory service — `brands` endpoints

Area: Brands

- `GET /v1/brands` → BrandController@index
- `POST /v1/brands` → BrandController@store
- `GET /v1/brands/{id}` → BrandController@show
- `PUT/PATCH /v1/brands/{id}` → BrandController@update
- `DELETE /v1/brands/{id}` → BrandController@destroy

### Inventory service — `consumable-issues` endpoints

Area: against (evaluation-service calls this with the clinician's token).

- `POST /v1/consumable-issues` → ConsumableIssueController@sync (needs permission:consumables.record|consumables.update|consumables.delete|evaluation.manage|inventory.adjust_stock|procedures.perform)

### Inventory service — `dha` endpoints

Area: Any prescriber: the link made as a prescription is saved.

- `GET /v1/dha/catalogue` → ProductDhaController@search
- `GET /v1/dha/reference/{type}` → ProductDhaController@reference
- `POST /v1/dha/resolve` → ProductDhaController@resolve

### Inventory service — `direct-orders` endpoints

Area: stays as a fallback for roles that hold it for other reasons.

- `POST /v1/direct-orders/{id}/receive` → DirectOrderController@receive (needs permission:inventory.direct_orders.manage|inventory.manage)
- `POST /v1/direct-orders/{id}/complete` → DirectOrderController@complete (needs permission:inventory.direct_orders.manage|inventory.manage)
- `POST /v1/direct-orders/{id}/cancel` → DirectOrderController@cancel (needs permission:inventory.direct_orders.manage|inventory.manage)
- `POST /v1/direct-orders/{id}/items` → DirectOrderController@addItem (needs permission:inventory.direct_orders.manage|inventory.manage)
- `DELETE /v1/direct-orders/{id}/items/{itemId}` → DirectOrderController@removeItem (needs permission:inventory.direct_orders.manage|inventory.manage)
- `POST /v1/direct-orders/{id}/record-payment` → DirectOrderController@recordPayment (needs permission:inventory.direct_orders.manage|inventory.manage)
- `GET /v1/direct-orders` → DirectOrderController@index
- `POST /v1/direct-orders` → DirectOrderController@store
- `GET /v1/direct-orders/{id}` → DirectOrderController@show
- `PUT/PATCH /v1/direct-orders/{id}` → DirectOrderController@update
- `DELETE /v1/direct-orders/{id}` → DirectOrderController@destroy

### Inventory service — `fieldmatch` endpoints

Area: Username + password (issued in hmis-manage) → bearer token for the routes below.

- `POST /v2/fieldmatch/login` → FieldMatchAuthController@login
- `POST /v2/fieldmatch/logout` → FieldMatchAuthController@logout
- `GET /v2/fieldmatch/stores` → FieldMatchController@stores
- `GET /v2/fieldmatch/catalogue` → FieldMatchController@catalogue
- `POST /v2/fieldmatch/price-imports` → FieldMatchController@priceImports

### Inventory service — `gateway` endpoints

Area: Model Gateway — cross-tenant read/seed/copy endpoint

- `POST /v1/gateway` → ModelGatewayController@handle

### Inventory service — `get` endpoints

Area: V2-compatible store product search used by inpatient and evaluation modules

- `GET /v1/get/store/products` → ProductController@getStoreProducts

### Inventory service — `goods-received-notes` endpoints

Area: Goods Received Notes

- `POST /v1/goods-received-notes/{id}/approve` → GoodsReceivedNoteController@approve (needs permission:inventory.grn.approve)
- `POST /v1/goods-received-notes/{id}/inspect` → GoodsReceivedNoteController@inspect (needs permission:inventory.grn.update|inventory.grn.approve)
- `POST /v1/goods-received-notes/{id}/reject` → GoodsReceivedNoteController@reject (needs permission:inventory.grn.reject)
- `GET /v1/goods-received-notes/{id}/print` → GoodsReceivedNoteController@generateGrn
- `GET /v1/goods-received-notes` → GoodsReceivedNoteController@index
- `POST /v1/goods-received-notes` → GoodsReceivedNoteController@store
- `GET /v1/goods-received-notes/{id}` → GoodsReceivedNoteController@show
- `PUT/PATCH /v1/goods-received-notes/{id}` → GoodsReceivedNoteController@update
- `DELETE /v1/goods-received-notes/{id}` → GoodsReceivedNoteController@destroy

### Inventory service — `health` endpoints

Area: Health check endpoint (unversioned, public)

- `GET /health` → HealthCheckController@check

### Inventory service — `internal` endpoints

Area: services.internal.* credential, not hmis_manage's).

- `POST /v1/internal/organizations/{id}/seed-product-catalogue` → InventoryCatalogueController@internalSeedFullCatalogue

### Inventory service — `inventory` endpoints

Area: Legacy compatibility endpoint

- `POST /v1/inventory/update-prescription-description` → ResourceController@updatePrescriptionDescription

### Inventory service — `medications` endpoints

Area: user session, only a registered signed app.

- `GET /v2/medications` → MedicationController@index
- `GET /v2/medications/{id}` → MedicationController@show
- `POST /v2/medications` → MedicationController@store

### Inventory service — `options` endpoints

Area: Form Options; a payment decision already made and authorized in finance-service.

- `GET /v1/options/formulation` → ProductController@formulationOptions
- `GET /v1/options/formulations` → OptionsController@formulations
- `GET /v1/options/label-types` → OptionsController@labelTypes
- `GET /v1/options` → OptionsController@index
- `GET /v1/options/{group}` → OptionsController@show

### Inventory service — `payment-modes` endpoints

Area: Payment Terms

- `GET /v1/payment-modes` → PaymentModeController@index
- `POST /v1/payment-modes` → PaymentModeController@store
- `GET /v1/payment-modes/{id}` → PaymentModeController@show
- `PUT/PATCH /v1/payment-modes/{id}` → PaymentModeController@update
- `DELETE /v1/payment-modes/{id}` → PaymentModeController@destroy

### Inventory service — `payment-terms` endpoints

Area: Payment Terms

- `POST /v1/payment-terms/{id}/activate` → PaymentTermController@activate (needs permission:inventory.manage)
- `POST /v1/payment-terms/{id}/deactivate` → PaymentTermController@deactivate (needs permission:inventory.manage)
- `GET /v1/payment-terms` → PaymentTermController@index
- `POST /v1/payment-terms` → PaymentTermController@store
- `GET /v1/payment-terms/{id}` → PaymentTermController@show
- `PUT/PATCH /v1/payment-terms/{id}` → PaymentTermController@update
- `DELETE /v1/payment-terms/{id}` → PaymentTermController@destroy

### Inventory service — `product-categories` endpoints

Area: Product Categories

- `POST /v1/product-categories/bulk-update-consumable` → ProductCategoryController@bulkUpdateConsumable (needs permission:inventory.categories.update|inventory.manage)
- `POST /v1/product-categories/bulk-update-drug` → ProductCategoryController@bulkUpdateDrug (needs permission:inventory.categories.update|inventory.manage)
- `GET /v1/product-categories` → ProductCategoryController@index
- `POST /v1/product-categories` → ProductCategoryController@store
- `GET /v1/product-categories/{id}` → ProductCategoryController@show
- `PUT/PATCH /v1/product-categories/{id}` → ProductCategoryController@update
- `DELETE /v1/product-categories/{id}` → ProductCategoryController@destroy

### Inventory service — `product-variants` endpoints

Area: Product Variants

- `GET /v1/product-variants/product/{productId}` → ProductVariantController@byProduct
- `POST /v1/product-variants/{id}/set-default` → ProductVariantController@setAsDefault (needs permission:inventory.items.update|inventory.manage)
- `GET /v1/product-variants` → ProductVariantController@index
- `POST /v1/product-variants` → ProductVariantController@store
- `GET /v1/product-variants/{id}` → ProductVariantController@show
- `PUT/PATCH /v1/product-variants/{id}` → ProductVariantController@update
- `DELETE /v1/product-variants/{id}` → ProductVariantController@destroy

### Inventory service — `products` endpoints

Area: Any prescriber: the link made as a prescription is saved.; Any prescriber: the price a prescription is saved at.; Backward-compatible public product search routes without /v1 prefix; DHA national product catalogue: match stocked drugs, confirm per drug.; Products; that silently never worked.

- `GET /products/search` → ProductController@search
- `GET /products/catalog-search` → ProductController@catalogSearch
- `GET /v1/products/export` → ProductController@export
- `GET /v1/products/search` → ProductController@search
- `GET /v1/products/catalog-search` → ProductController@catalogSearch
- `GET /v1/products/by-store/{storeId}` → ProductController@byStore
- `GET /v1/products/{id}/stock-info` → ProductController@stockInfo
- `POST /v1/products/bulk-import` → ProductController@bulkImport (needs permission:inventory.items.import|inventory.manage)
- `POST /v1/products/bulk-update-prices` → ProductController@bulkUpdatePrices (needs permission:inventory.items.update|inventory.manage)
- `GET /v1/products/{product}/price` → ProductController@storePrice
- `POST /v1/products/{product}/store-price` → ProductController@updateStorePrice (needs permission:inventory.items.update|inventory.manage)
- `GET /v1/products/bulk-prices-export` → ProductController@bulkPricesExport (needs permission:inventory.items.update|inventory.manage)
- `POST /v1/products/bulk-prices-import` → ProductController@bulkPricesImport (needs permission:inventory.items.update|inventory.manage)
- `GET /v1/products/store-price-overrides` → ProductController@storePriceOverrides (needs permission:inventory.items.update|inventory.manage)
- `POST /v1/products/dha/suggest` → ProductDhaController@suggest (needs permission:inventory.items.update|inventory.manage)
- `POST /v1/products/dha/reset` → ProductDhaController@reset (needs permission:inventory.items.update|inventory.manage)
- `POST /v1/products/dha/confirm-all` → ProductDhaController@confirmAll (needs permission:inventory.items.update|inventory.manage)
- `POST /v1/products/{product}/dha/link` → ProductDhaController@link
- `POST /v1/products/{product}/dha` → ProductDhaController@confirm (needs permission:inventory.items.update|inventory.manage)
- `DELETE /v1/products/{product}/dha/suggestion` → ProductDhaController@dismiss (needs permission:inventory.items.update|inventory.manage)
- `DELETE /v1/products/{product}/dha` → ProductDhaController@clear (needs permission:inventory.items.update|inventory.manage)
- `GET /v1/products` → ProductController@index
- `POST /v1/products` → ProductController@store
- `GET /v1/products/{id}` → ProductController@show
- `PUT/PATCH /v1/products/{id}` → ProductController@update
- `DELETE /v1/products/{id}` → ProductController@destroy

### Inventory service — `purchase-orders` endpoints

Area: Purchase Orders

- `POST /v1/purchase-orders/{id}/submit` → PurchaseOrderController@submit (needs permission:inventory.purchase_orders.submit|inventory.purchase_orders.create)
- `POST /v1/purchase-orders/{id}/approve` → PurchaseOrderController@approve
- `POST /v1/purchase-orders/{id}/reject` → PurchaseOrderController@reject
- `POST /v1/purchase-orders/{id}/cancel` → PurchaseOrderController@cancel (needs permission:inventory.purchase_orders.cancel|inventory.purchase_orders.create|inventory.approve_purchase_orders)
- `POST /v1/purchase-orders/{id}/send` → PurchaseOrderController@send (needs permission:inventory.purchase_orders.submit|inventory.purchase_orders.create)
- `POST /v1/purchase-orders/{id}/supplier-accept` → PurchaseOrderController@supplierAccept (needs permission:inventory.purchase_orders.receive|inventory.approve_purchase_orders)
- `POST /v1/purchase-orders/{id}/supplier-reject` → PurchaseOrderController@supplierReject (needs permission:inventory.purchase_orders.receive|inventory.approve_purchase_orders)
- `POST /v1/purchase-orders/{id}/record-payment` → PurchaseOrderController@recordPayment (needs permission:inventory.purchase_orders.receive|inventory.approve_purchase_orders)
- `GET /v1/purchase-orders/{id}/lpo` → PurchaseOrderController@generateLpo
- `POST /v1/purchase-orders/{id}/lpo/email` → PurchaseOrderController@emailLpo (needs permission:inventory.purchase_orders.submit|inventory.purchase_orders.create)
- `GET /v1/purchase-orders` → PurchaseOrderController@index
- `POST /v1/purchase-orders` → PurchaseOrderController@store
- `GET /v1/purchase-orders/{id}` → PurchaseOrderController@show
- `PUT/PATCH /v1/purchase-orders/{id}` → PurchaseOrderController@update
- `DELETE /v1/purchase-orders/{id}` → PurchaseOrderController@destroy

### Inventory service — `requisition-approvals` endpoints

Area: Requisition Approvals

- `GET /v1/requisition-approvals/requisition/{requisitionId}` → RequisitionApprovalController@index
- `GET /v1/requisition-approvals/pending` → RequisitionApprovalController@pending
- `GET /v1/requisition-approvals/statistics` → RequisitionApprovalController@statistics
- `GET /v1/requisition-approvals/my-approvals` → RequisitionApprovalController@myApprovals

### Inventory service — `requisitions` endpoints

Area: Requisitions

- `POST /v1/requisitions/{id}/submit` → RequisitionController@submit (needs permission:inventory.requisitions.submit|inventory.requisitions.create)
- `POST /v1/requisitions/{id}/approve` → RequisitionController@approve
- `POST /v1/requisitions/{id}/force-approve` → RequisitionController@forceApprove
- `POST /v1/requisitions/{id}/reject` → RequisitionController@reject
- `POST /v1/requisitions/{id}/fulfill` → RequisitionController@fulfill
- `POST /v1/requisitions/{id}/assign-supplier` → RequisitionController@assignSupplier (needs permission:inventory.requisitions.update|inventory.approve_requisitions)
- `POST /v1/requisitions/{id}/convert-to-po` → RequisitionController@convertToPo
- `GET /v1/requisitions/{id}/print` → RequisitionController@generateDocument
- `GET /v1/requisitions/{id}/rfq` → RequisitionController@generateRfq
- `POST /v1/requisitions/{id}/rfq/email` → RequisitionController@emailRfq (needs permission:inventory.requisitions.update|inventory.approve_requisitions)
- `GET /v1/requisitions` → RequisitionController@index
- `POST /v1/requisitions` → RequisitionController@store
- `GET /v1/requisitions/{id}` → RequisitionController@show
- `PUT/PATCH /v1/requisitions/{id}` → RequisitionController@update
- `DELETE /v1/requisitions/{id}` → RequisitionController@destroy

### Inventory service — `sales` endpoints

Area: (Cash/Insurance Workbench) — see itemsStatus() docblock.; ===================================================================; Cash Workbench / Insurance Workbench to track and settle.; a payment decision already made and authorized in finance-service.; a previously-saved Pending sale later (plain "Save" or credit).; per-line-item dispense tracking, station + date-range scoped.

- `GET /v1/sales` → SaleController@index
- `POST /v1/sales/checkout` → SaleController@checkout (needs permission:pos.sales.create)
- `GET /v1/sales/items` → SaleController@items
- `GET /v1/sales/items/status` → SaleController@itemsStatus
- `POST /v1/sales/items/dispense` → SaleController@dispenseItems
- `GET /v1/sales/pending-pharmacy` → SaleController@pendingPharmacySales (needs permission:pos.sales.receive_payment)
- `POST /v1/sales/{id}/process-payment` → SaleController@processPayment (needs permission:pos.sales.receive_payment)
- `POST /v1/sales/{id}/mark-paid` → SaleController@markPaid
- `GET /v1/sales/shifts/{shiftId}` → SaleController@shiftSales
- `GET /v1/sales/shift-financials/{shiftId}` → SaleController@shiftFinancials
- `GET /v1/sales/{id}` → SaleController@show
- `POST /v1/sales/{id}/void` → SaleController@void (needs permission:pos.manage)

### Inventory service — `serial-numbers` endpoints

Area: Serial Numbers

- `POST /v1/serial-numbers/find` → SerialNumberController@findBySerial
- `POST /v1/serial-numbers/{id}/issue` → SerialNumberController@issue (needs permission:inventory.manage_stock|inventory.stock.update)
- `POST /v1/serial-numbers/{id}/return` → SerialNumberController@returnToStock (needs permission:inventory.manage_stock|inventory.stock.update)
- `POST /v1/serial-numbers/{id}/mark-damaged` → SerialNumberController@markAsDamaged (needs permission:inventory.manage_stock|inventory.stock.update)
- `GET /v1/serial-numbers/status/{status}` → SerialNumberController@byStatus
- `GET /v1/serial-numbers/under-warranty` → SerialNumberController@underWarranty
- `GET /v1/serial-numbers/warranty-expiring` → SerialNumberController@warrantyExpiringSoon
- `GET /v1/serial-numbers/needing-inspection` → SerialNumberController@needingInspection
- `GET /v1/serial-numbers` → SerialNumberController@index
- `POST /v1/serial-numbers` → SerialNumberController@store
- `GET /v1/serial-numbers/{id}` → SerialNumberController@show
- `PUT/PATCH /v1/serial-numbers/{id}` → SerialNumberController@update
- `DELETE /v1/serial-numbers/{id}` → SerialNumberController@destroy

### Inventory service — `service` endpoints

Area: api/service/... instead and every client calling it would 404.

- `POST /v1/service/organizations/{id}/seed-reference-data` → InventoryCatalogueController@seedReferenceData

### Inventory service — `stock` endpoints

Area: Stock

- `GET /v1/stock` → StockController@index
- `POST /v1/stock/dispensable` → StockController@dispensable
- `GET /v1/stock/{id}` → StockController@show
- `GET /v1/stock/by-product/{productId}` → StockController@byProduct
- `GET /v1/stock/by-store/{storeId}` → StockController@byStore
- `GET /v1/stock/by-store/{storeId}/export` → StockController@byStoreStream
- `GET /v1/stock/low-stock/list` → StockController@lowStock
- `GET /v1/stock/summary/overview` → StockController@summary
- `GET /v1/stock/summary/by-store` → StockController@summaryByStore

### Inventory service — `stock-adjustments` endpoints

Area: procedure flows post stock deductions here with their own token.

- `POST /v1/stock-adjustments` → StockAdjustmentController@create (needs permission:inventory.adjust_stock|inventory.stock.adjust|prescriptions.dispense|consumables.record|procedures.perform)
- `POST /v1/stock-adjustments/{id}/approve` → StockAdjustmentController@approve
- `GET /v1/stock-adjustments/pending` → StockAdjustmentController@pending

### Inventory service — `stock-movements` endpoints

Area: Finance staff reverse stock on invoice void/refund with their own token.; Stock Movements

- `GET /v1/stock-movements/by-product/{productId}` → StockMovementController@byProduct
- `GET /v1/stock-movements/by-store/{storeId}` → StockMovementController@byStore
- `GET /v1/stock-movements/export` → StockMovementController@export
- `GET /v1/stock-movements/summary` → StockMovementController@summary
- `GET /v1/stock-movements/trend` → StockMovementController@trend
- `POST /v1/stock-movements/reverse` → StockMovementController@reverse (needs permission:inventory.adjust_stock|inventory.manage_stock|payments.reverse|receipts.void|finance.manage)
- `GET /v1/stock-movements` → StockMovementController@index
- `GET /v1/stock-movements/{id}` → StockMovementController@show

### Inventory service — `stock-takes` endpoints

Area: Stock Takes (require manage_stock_takes for write operations)

- `POST /v1/stock-takes/{id}/count` → StockTakeController@recordCounts
- `POST /v1/stock-takes/{id}/submit` → StockTakeController@submit
- `POST /v1/stock-takes/{id}/cancel` → StockTakeController@cancel
- `POST /v1/stock-takes/{id}/schedule` → StockTakeController@schedule
- `GET /v1/stock-takes` → StockTakeController@index
- `POST /v1/stock-takes` → StockTakeController@store
- `GET /v1/stock-takes/{id}` → StockTakeController@show

### Inventory service — `stock-transfers` endpoints

Area: Stock Transfers

- `POST /v1/stock-transfers` → StockTransferController@create (needs permission:inventory.transfer_stock|inventory.stock.transfer)
- `GET /v1/stock-transfers/history` → StockTransferController@history
- `GET /v1/stock-transfers/{id}` → StockTransferController@show

### Inventory service — `store-orders` endpoints

Area: Store Orders (Internal Transfers)

- `POST /v1/store-orders/{id}/accept` → StoreOrderController@accept (needs permission:inventory.manage_stock|inventory.transfer_stock)
- `POST /v1/store-orders/{id}/dispatch` → StoreOrderController@dispatch (needs permission:inventory.manage_stock|inventory.transfer_stock)
- `POST /v1/store-orders/{id}/receive` → StoreOrderController@receive (needs permission:inventory.manage_stock|inventory.transfer_stock)
- `POST /v1/store-orders/{id}/cancel` → StoreOrderController@cancel (needs permission:inventory.manage_stock|inventory.transfer_stock)
- `GET /v1/store-orders` → StoreOrderController@index
- `POST /v1/store-orders` → StoreOrderController@store
- `GET /v1/store-orders/{id}` → StoreOrderController@show
- `PUT/PATCH /v1/store-orders/{id}` → StoreOrderController@update
- `DELETE /v1/store-orders/{id}` → StoreOrderController@destroy

### Inventory service — `stores` endpoints

Area: Stores/Warehouses

- `GET /v1/stores/stream` → StoreController@stream
- `GET /v1/stores` → StoreController@index
- `POST /v1/stores` → StoreController@store
- `GET /v1/stores/{id}` → StoreController@show
- `PUT/PATCH /v1/stores/{id}` → StoreController@update
- `DELETE /v1/stores/{id}` → StoreController@destroy

### Inventory service — `suppliers` endpoints

Area: Suppliers

- `GET /v1/suppliers/search` → SupplierController@search
- `POST /v1/suppliers/bulk-update` → SupplierController@bulkUpdate (needs permission:inventory.suppliers.update|inventory.suppliers.create)
- `GET /v1/suppliers/{id}/payment-history` → SupplierController@paymentHistory
- `GET /v1/suppliers/{id}/payment-log` → SupplierController@paymentLog
- `GET /v1/suppliers/{id}/statement` → SupplierController@generateStatement
- `GET /v1/suppliers/{id}/orders/{orderType}/{orderId}/invoice` → SupplierController@generateInvoice
- `GET /v1/suppliers/{id}/payments/{paymentId}/voucher` → SupplierController@generatePaymentVoucher
- `GET /v1/suppliers/{id}/payments/{paymentId}/receipt` → SupplierController@generateReceipt
- `GET /v1/suppliers` → SupplierController@index
- `POST /v1/suppliers` → SupplierController@store
- `GET /v1/suppliers/{id}` → SupplierController@show
- `PUT/PATCH /v1/suppliers/{id}` → SupplierController@update
- `DELETE /v1/suppliers/{id}` → SupplierController@destroy

### Inventory service — `tax-categories` endpoints

Area: Tax Categories

- `GET /v1/tax-categories` → TaxCategoryController@index
- `POST /v1/tax-categories` → TaxCategoryController@store
- `GET /v1/tax-categories/{id}` → TaxCategoryController@show
- `PUT/PATCH /v1/tax-categories/{id}` → TaxCategoryController@update
- `DELETE /v1/tax-categories/{id}` → TaxCategoryController@destroy

### Inventory service — `units` endpoints

Area: Units of Measurement

- `GET /v1/units` → UnitController@index
- `POST /v1/units` → UnitController@store
- `GET /v1/units/{id}` → UnitController@show
- `PUT/PATCH /v1/units/{id}` → UnitController@update
- `DELETE /v1/units/{id}` → UnitController@destroy

## Morgue service API (morgue-service)

Covers: bodies, storage, releases, mortuary charges. 40 endpoints.

### Morgue service — `gateway` endpoints

Area: same path in every service by the migration harness.

- `POST /v1/gateway` → ModelGatewayController@handle

### Morgue service — `health` endpoints

- `GET /health` → HealthCheckController@check

### Morgue service — `morgue` endpoints

Area: -- Procedures ------------------------------------------------------; -- Property held with a body ---------------------------------------; -- The rate card ---------------------------------------------------; after the fact is exactly what gets asked about later.; because receiving IS the admission.; finance or waived with a recorded reason, and nothing else.

- `GET /v1/morgue/bodies` → BodyController@index (needs permission:morgue.view|morgue.admit)
- `GET /v1/morgue/bodies/{body}` → BodyController@show (needs permission:morgue.view|morgue.admit)
- `POST /v1/morgue/bodies` → BodyController@store (needs permission:morgue.admit)
- `PATCH /v1/morgue/bodies/{body}` → BodyController@update (needs permission:morgue.admit)
- `POST /v1/morgue/bodies/{body}/placements` → BodyController@place (needs permission:morgue.storage.manage|morgue.admit)
- `POST /v1/morgue/bodies/{body}/remove` → BodyController@removeFromSpace (needs permission:morgue.storage.manage)
- `GET /v1/morgue/fridges` → StorageController@index (needs permission:morgue.view|morgue.admit)
- `GET /v1/morgue/spaces/available` → StorageController@available (needs permission:morgue.view|morgue.admit)
- `POST /v1/morgue/fridges` → StorageController@storeFridge (needs permission:morgue.manage)
- `POST /v1/morgue/fridges/{fridge}/spaces` → StorageController@storeSpace (needs permission:morgue.manage)
- `PATCH /v1/morgue/spaces/{space}` → StorageController@updateSpace (needs permission:morgue.storage.manage|morgue.manage)
- `GET /v1/morgue/releases` → ReleaseController@index (needs permission:morgue.view|morgue.release.raise|morgue.release.approve)
- `GET /v1/morgue/releases/{release}` → ReleaseController@show (needs permission:morgue.view|morgue.release.raise|morgue.release.approve)
- `POST /v1/morgue/bodies/{body}/releases` → ReleaseController@store (needs permission:morgue.release.raise)
- `POST /v1/morgue/releases/{release}/approve` → ReleaseController@approve (needs permission:morgue.release.approve)
- `POST /v1/morgue/releases/{release}/complete` → ReleaseController@complete (needs permission:morgue.release.raise|morgue.release.approve)
- `POST /v1/morgue/releases/{release}/cancel` → ReleaseController@cancel (needs permission:morgue.release.raise|morgue.release.approve)
- `GET /v1/morgue/bodies/{body}/effects` → EffectController@index (needs permission:morgue.view|morgue.admit)
- `POST /v1/morgue/bodies/{body}/effects` → EffectController@store (needs permission:morgue.admit|morgue.release.raise)
- `POST /v1/morgue/effects/{effect}/release` → EffectController@release (needs permission:morgue.release.raise)
- `GET /v1/morgue/bodies/{body}/procedures` → ChargeController@procedures (needs permission:morgue.view|morgue.admit)
- `POST /v1/morgue/bodies/{body}/procedures` → ChargeController@storeProcedure (needs permission:morgue.procedure.record)
- `GET /v1/morgue/bodies/{body}/charges` → ChargeController@index (needs permission:morgue.charges.view|morgue.view)
- `POST /v1/morgue/bodies/{body}/charges` → ChargeController@store (needs permission:morgue.charges.manage)
- `POST /v1/morgue/charges/{charge}/waive` → ChargeController@waive (needs permission:morgue.charges.waive)
- `GET /v1/morgue/rates` → RateController@index (needs permission:morgue.charges.view|morgue.view|morgue.manage)
- `POST /v1/morgue/rates` → RateController@store (needs permission:morgue.rates.manage)
- `PATCH /v1/morgue/rates/{rate}` → RateController@update (needs permission:morgue.rates.manage)
- `GET /v1/morgue/finance/billable` → MorgueFinanceController@billable (needs permission:morgue.charges.view|finance.payments.view|payments.view)
- `POST /v1/morgue/bodies/{body}/mark-billed` → MorgueFinanceController@markBilled (needs permission:morgue.charges.manage|finance.payments.create|payments.create)
- `GET /v1/morgue/intakes` → IntakeController@index (needs permission:morgue.view|morgue.admit|morgue.intake.request)
- `GET /v1/morgue/intakes/{intake}` → IntakeController@show (needs permission:morgue.view|morgue.admit|morgue.intake.request)
- `POST /v1/morgue/intakes` → IntakeController@store (needs permission:morgue.intake.request|morgue.admit)
- `POST /v1/morgue/intakes/{intake}/receive` → IntakeController@receive (needs permission:morgue.admit)
- `POST /v1/morgue/intakes/{intake}/cancel` → IntakeController@cancel (needs permission:morgue.admit|morgue.intake.request)
- `GET /v1/morgue/unclaimed` → UnclaimedController@index (needs permission:morgue.view|morgue.admit)
- `GET /v1/morgue/bodies/{body}/unclaimed-actions` → UnclaimedController@actions (needs permission:morgue.view|morgue.admit)
- `POST /v1/morgue/bodies/{body}/unclaimed-actions` → UnclaimedController@store (needs permission:morgue.unclaimed.manage|morgue.admit)

## Oncology service API (oncology-service)

Covers: screening, diagnosis, treatment plans, chemo cycles, palliative care. 97 endpoints.

### Oncology service — `gateway` endpoints

Area: at the same path, and oncology is not a special case to it.

- `POST /v1/gateway` → ModelGatewayController@handle

### Oncology service — `health` endpoints

- `GET /health` → HealthCheckController@check

### Oncology service — `oncology` endpoints

Area: Approval is the freeze. Oncologist only.; and a database CHECK constraint. Three layers, none of them Vue.; attended, and it would still reach the queues and the bill.; controller — so they cannot come from the plans listing.; diagnosis, so it needs the diagnosis-write permission too.; it. See ScreeningOrderController.

- `GET /v1/oncology/registry/tumour-streams` → RegistryController@tumourStreams
- `GET /v1/oncology/registry/protocols` → RegistryController@protocols
- `GET /v1/oncology/registry/protocols/{protocol}` → RegistryController@protocol
- `GET /v1/oncology/registry/screening-programmes` → RegistryController@screeningProgrammes
- `GET /v1/oncology/registry/staging-systems` → RegistryController@stagingSystems
- `GET /v1/oncology/registry/icd-o` → RegistryController@icdO
- `GET /v1/oncology/registry/response-criteria` → RegistryController@responseCriteria
- `GET /v1/oncology/registry/tariffs` → RegistryController@tariffs
- `GET /v1/oncology/registry/cumulative-dose-limits` → RegistryController@cumulativeDoseLimits
- `GET /v1/oncology/registry/ctcae-terms` → RegistryController@ctcaeTerms
- `GET /v1/oncology/screenings/eligibility` → ScreeningController@eligibility
- `GET /v1/oncology/screenings/summary` → ScreeningController@summary
- `GET /v1/oncology/screenings/due` → ScreeningController@due
- `GET /v1/oncology/screenings/due/summary` → ScreeningController@dueSummary
- `GET /v1/oncology/screenings` → ScreeningController@index
- `GET /v1/oncology/screenings/{screening}` → ScreeningController@show
- `POST /v1/oncology/screenings` → ScreeningController@store (needs permission:oncology.screening.record)
- `POST /v1/oncology/screening-orders` → ScreeningOrderController@store
- `GET /v1/oncology/screening-orders` → ScreeningOrderController@index (needs permission:oncology.screening.record|oncology.view)
- `POST /v1/oncology/screening-orders/{order}/cancel` → ScreeningOrderController@cancel (needs permission:oncology.screening.record)
- `GET /v1/oncology/referrals` → ReferralController@index
- `GET /v1/oncology/referrals/summary` → ReferralController@summary
- `GET /v1/oncology/referrals/{referral}` → ReferralController@show
- `POST /v1/oncology/referrals` → ReferralController@store (needs permission:oncology.screening.record)
- `POST /v1/oncology/referrals/{referral}/outcome` → ReferralController@outcome (needs permission:oncology.diagnosis.write|oncology.screening.record)
- `POST /v1/oncology/referrals/{referral}/attempts` → ReferralController@attempt (needs permission:oncology.screening.record)
- `POST /v1/oncology/referrals/{referral}/lost` → ReferralController@markLost (needs permission:oncology.screening.record)
- `POST /v1/oncology/referrals/{referral}/assign-chp` → ReferralController@assignChp (needs permission:oncology.screening.record)
- `GET /v1/oncology/diagnoses` → DiagnosisController@index
- `GET /v1/oncology/diagnoses/{diagnosis}` → DiagnosisController@show
- `POST /v1/oncology/diagnoses` → DiagnosisController@store (needs permission:oncology.diagnosis.write)
- `POST /v1/oncology/diagnoses/{diagnosis}/revisions` → DiagnosisController@revise (needs permission:oncology.diagnosis.write)
- `GET /v1/oncology/diagnoses/{diagnosis}/stagings` → DiagnosisController@stagings
- `POST /v1/oncology/diagnoses/{diagnosis}/stagings` → DiagnosisController@stage (needs permission:oncology.staging.write)
- `POST /v1/oncology/staging/derive` → DiagnosisController@deriveStageGroup
- `GET /v1/oncology/mdt/meetings` → MdtController@meetings
- `POST /v1/oncology/mdt/meetings` → MdtController@storeMeeting (needs permission:oncology.mdt.decide)
- `GET /v1/oncology/mdt/meetings/{meeting}` → MdtController@showMeeting
- `PATCH /v1/oncology/mdt/meetings/{meeting}/attendance` → MdtController@attendance (needs permission:oncology.mdt.decide)
- `POST /v1/oncology/mdt/meetings/{meeting}/cases` → MdtController@storeCase (needs permission:oncology.mdt.decide)
- `GET /v1/oncology/mdt/cases/{case}` → MdtController@showCase
- `POST /v1/oncology/mdt/cases/{case}/decision` → MdtController@decide (needs permission:oncology.mdt.decide)
- `GET /v1/oncology/treatment-plans` → TreatmentPlanController@index
- `GET /v1/oncology/treatment-plans/{plan}` → TreatmentPlanController@show
- `GET /v1/oncology/treatment-plans/{plan}/projection` → TreatmentPlanController@projection
- `GET /v1/oncology/treatment-plans/{plan}/stopping-rules` → TreatmentPlanController@stoppingRules
- `GET /v1/oncology/treatment-plans/{plan}/toxicity-grid` → ClinicalRecordController@toxicityGrid
- `POST /v1/oncology/treatment-plans` → TreatmentPlanController@store (needs permission:oncology.plan.approve|oncology.diagnosis.write)
- `POST /v1/oncology/treatment-plans/{plan}/approve` → TreatmentPlanController@approve (needs permission:oncology.plan.approve)
- `POST /v1/oncology/treatment-plans/{plan}/close` → TreatmentPlanController@close (needs permission:oncology.plan.approve)
- `POST /v1/oncology/treatment-plans/{plan}/suspend` → TreatmentPlanController@suspend (needs permission:oncology.plan.approve)
- `POST /v1/oncology/treatment-plans/{plan}/reproject` → PlanCycleController@reproject (needs permission:oncology.cycle.open|oncology.plan.approve)
- `POST /v1/oncology/treatment-plans/{plan}/care-plan` → PalliativeController@materialise (needs permission:oncology.plan.approve)
- `GET /v1/oncology/cycles/{cycle}` → PlanCycleController@show
- `GET /v1/oncology/cycles/{cycle}/gate` → PlanCycleController@gate
- `GET /v1/oncology/cycles/{cycle}/dose-preview` → PlanCycleController@dosePreview
- `POST /v1/oncology/cycles/{cycle}/open` → PlanCycleController@open (needs permission:oncology.cycle.open)
- `POST /v1/oncology/cycles/{cycle}/defer` → PlanCycleController@defer (needs permission:oncology.cycle.open)
- `POST /v1/oncology/cycles/{cycle}/confirm-stock` → PlanCycleController@confirmStock (needs permission:oncology.cycle.open|oncology.cytotoxic.handle)
- `GET /v1/oncology/cycles/{cycle}/administrations` → CycleAdministrationController@index
- `POST /v1/oncology/cycles/{cycle}/administrations` → CycleAdministrationController@store (needs permission:oncology.cytotoxic.handle)
- `POST /v1/oncology/administrations/{administration}/revisions` → CycleAdministrationController@revise (needs permission:oncology.cytotoxic.handle)
- `GET /v1/oncology/observations` → ClinicalRecordController@observations
- `POST /v1/oncology/observations` → ClinicalRecordController@storeObservation (needs permission:oncology.observations.write|oncology.toxicity.write)
- `POST /v1/oncology/cycles/{cycle}/observations` → ClinicalRecordController@storeInfusionObservation (needs permission:oncology.observations.write|oncology.toxicity.write)
- `GET /v1/oncology/toxicities` → ClinicalRecordController@toxicities
- `POST /v1/oncology/toxicities` → ClinicalRecordController@storeToxicity (needs permission:oncology.toxicity.write)
- `GET /v1/oncology/response-assessments` → ClinicalRecordController@responseAssessments
- `POST /v1/oncology/response-assessments` → ClinicalRecordController@storeResponseAssessment (needs permission:oncology.toxicity.write|oncology.diagnosis.write)
- `GET /v1/oncology/benefit/{patient}` → BenefitController@show (needs permission:oncology.benefit.view)
- `GET /v1/oncology/benefit/{patient}/lines` → BenefitController@lines (needs permission:oncology.benefit.view)
- `GET /v1/oncology/benefit/{patient}/projection` → BenefitController@projection (needs permission:oncology.benefit.view)
- `POST /v1/oncology/benefit/{patient}/refresh` → BenefitController@refresh (needs permission:oncology.benefit.view)
- `POST /v1/oncology/benefit/lines/submit` → BenefitController@submitAccrued (needs permission:oncology.benefit.view)
- `GET /v1/oncology/patients/{patient}/charting-visit` → ClinicBookingController@chartingVisit
- `POST /v1/oncology/patients/{patient}/charting-visit` → ClinicBookingController@startChartingVisit
- `GET /v1/oncology/clinic-bookings` → ClinicBookingController@index
- `POST /v1/oncology/clinic-bookings` → ClinicBookingController@store
- `POST /v1/oncology/clinic-bookings/{booking}/status` → ClinicBookingController@updateStatus
- `GET /v1/oncology/worklists/day-unit` → WorklistController@dayUnit
- `GET /v1/oncology/worklists/treatment-sessions` → WorklistController@treatmentSessions
- `POST /v1/oncology/worklists/treatment-sessions/{session}/status` → WorklistController@updateTreatmentSession (needs permission:oncology.screening.record|oncology.diagnosis.write)
- `GET /v1/oncology/worklists/tracing` → WorklistController@tracing
- `POST /v1/oncology/worklists/tracing/assign` → WorklistController@assignChp
- `POST /v1/oncology/worklists/tracing/{item}/outcome` → WorklistController@closeTracingItem
- `GET /v1/oncology/palliative/assessments` → PalliativeController@assessments
- `POST /v1/oncology/palliative/assessments` → PalliativeController@storeAssessment (needs permission:oncology.toxicity.write|oncology.observations.write)
- `GET /v1/oncology/palliative/pain-trend` → PalliativeController@painTrend
- `POST /v1/oncology/palliative/controlled-drugs` → PalliativeController@storeControlledDrug (needs permission:oncology.cytotoxic.handle|oncology.controlled_drugs.dispense)
- `GET /v1/oncology/palliative/controlled-drugs/reconciliation` → PalliativeController@controlledDrugReconciliation
- `GET /v1/oncology/care-plan-items` → PalliativeController@carePlanItems
- `PATCH /v1/oncology/care-plan-items/{item}` → PalliativeController@updateCarePlanItem
- `GET /v1/oncology/surveillance-schedules` → PalliativeController@surveillanceSchedules
- `GET /v1/oncology/patients/{patient}/summary` → OncologySummaryController@show
- `GET /v1/oncology/patients/{patient}/cross-facility` → OncologySummaryController@crossFacility

## Patient evaluation service API (patient-evaluation-service)

Covers: outpatient clinical work, triage, notes, lab, pharmacy, procedures, SHA claims. 807 endpoints.

### Patient evaluation service — `(root)` endpoints

Area: so they ride the same three routes.

- `GET /v1/{$form}-register`
- `GET /v1/{$form}/{visit}`

### Patient evaluation service — `admission` endpoints

Area: warn/block when patients are still checked in there.

- `POST /v1/admission/pre_entered/details`

### Patient evaluation service — `adverse-drug-reactions` endpoints

Area: wildcard so they aren't swallowed by it.

- `GET /v1/adverse-drug-reactions/patient/{patientId}` → AdverseDrugReactionController@index
- `POST /v1/adverse-drug-reactions` → AdverseDrugReactionController@store (needs permission:clinical_history.allergy|clinical_history.record|evaluation.manage)

### Patient evaluation service — `age-groups` endpoints

Area: Lab Settings CRUD

- `GET /v1/age-groups` → AgeGroupController@index
- `POST /v1/age-groups` → AgeGroupController@store
- `GET /v1/age-groups/{id}` → AgeGroupController@show
- `PUT/PATCH /v1/age-groups/{id}` → AgeGroupController@update
- `DELETE /v1/age-groups/{id}` → AgeGroupController@destroy

### Patient evaluation service — `all-procedures` endpoints

Area: POS Procedures - Get procedures with prices (similar to collabmed's get_pos_procedures)

- `GET /v1/all-procedures` → ProcedureController@getProcedures

### Patient evaluation service — `allergy-intolerances` endpoints

Area: brand-new routes with no existing caller to preserve compatibility for.; §6 Phase 3.

- `GET /v2/allergy-intolerances` → AllergyIntoleranceController@index
- `GET /v2/allergy-intolerances/{id}` → AllergyIntoleranceController@show
- `POST /v2/allergy-intolerances` → AllergyIntoleranceController@store

### Patient evaluation service — `anc-enrollments` endpoints

Area: before the '{ancEnrollment}' wildcard so they aren't swallowed by it.

- `GET /v1/anc-enrollments` → AncEnrollmentController@index
- `GET /v1/anc-enrollments/patient/{patientId}` → AncEnrollmentController@byPatient
- `POST /v1/anc-enrollments` → AncEnrollmentController@store (needs permission:clinical_history.record|evaluation.manage)
- `GET /v1/anc-enrollments/{ancEnrollment}/schedule-preview` → AncEnrollmentController@schedulePreview
- `GET /v1/anc-enrollments/{ancEnrollment}/schedules` → AncEnrollmentController@schedules
- `POST /v1/anc-enrollments/{ancEnrollment}/confirm` → AncEnrollmentController@confirm (needs permission:evaluation.manage)
- `PUT /v1/anc-enrollments/{ancEnrollment}/schedule` → AncEnrollmentController@updateSchedule (needs permission:evaluation.manage)
- `POST /v1/anc-enrollments/{ancEnrollment}/cancel` → AncEnrollmentController@cancel (needs permission:evaluation.manage)
- `GET /v1/anc-enrollments/{ancEnrollment}` → AncEnrollmentController@show

### Patient evaluation service — `anc-visit-schedules` endpoints

Area: as an {fpService} id.

- `POST /v1/anc-visit-schedules/{ancVisitSchedule}/check-in` → AncVisitScheduleController@checkIn (needs permission:evaluation.manage)

### Patient evaluation service — `bio-reference-ranges` endpoints

Area: captured by the resource's {bio_reference_range} show route.

- `GET /v1/bio-reference-ranges/csv-template` → BioReferenceRangeController@csvTemplate
- `POST /v1/bio-reference-ranges/csv-import` → BioReferenceRangeController@csvImport (needs permission:lab.setup|settings.manage)
- `GET /v1/bio-reference-ranges` → BioReferenceRangeController@index
- `POST /v1/bio-reference-ranges` → BioReferenceRangeController@store
- `GET /v1/bio-reference-ranges/{id}` → BioReferenceRangeController@show
- `PUT/PATCH /v1/bio-reference-ranges/{id}` → BioReferenceRangeController@update
- `DELETE /v1/bio-reference-ranges/{id}` → BioReferenceRangeController@destroy

### Patient evaluation service — `collection-methods` endpoints

Area: Sample types & collection methods — full CRUD

- `GET /v1/collection-methods` → SampleCollectionMethodController@index
- `POST /v1/collection-methods` → SampleCollectionMethodController@store
- `GET /v1/collection-methods/{id}` → SampleCollectionMethodController@show
- `PUT/PATCH /v1/collection-methods/{id}` → SampleCollectionMethodController@update
- `DELETE /v1/collection-methods/{id}` → SampleCollectionMethodController@destroy

### Patient evaluation service — `compositions` endpoints

Area: §6 Phase 3.

- `GET /v2/compositions/{patientId}` → CompositionController@show

### Patient evaluation service — `conditions` endpoints

Area: brand-new routes with no existing caller to preserve compatibility for.; §6 Phase 3.

- `GET /v2/conditions` → ConditionController@index
- `GET /v2/conditions/{id}` → ConditionController@show
- `POST /v2/conditions` → ConditionController@store

### Patient evaluation service — `consumables` endpoints

Area: the general update() route.

- `PUT /v1/consumables/{consumable}/billing-sync` → ConsumableController@markBillingSync (needs permission:finance.process_payments|billing.manage|consumables.update|consumables.record|evaluation.manage)
- `PATCH /v1/consumables/{consumable}/cancel` → ConsumableController@cancel (needs permission:consumables.update|consumables.record|evaluation.manage)
- `GET /v1/consumables` → ConsumableController@index
- `POST /v1/consumables` → ConsumableController@store
- `GET /v1/consumables/{id}` → ConsumableController@show
- `PUT/PATCH /v1/consumables/{id}` → ConsumableController@update
- `DELETE /v1/consumables/{id}` → ConsumableController@destroy

### Patient evaluation service — `counselling` endpoints

Area: Counselling — stored as JSON in a dedicated table (coming) or returns empty for now; TODO: persist to evaluation_counselling table when migration is added

- `GET /counselling`
- `POST /counselling`

### Patient evaluation service — `create-visit` endpoints

Area: Visit Management

- `POST /v1/create-visit` → VisitController@createVisit (needs permission:reception.visits.create|visits.create|evaluation.visits.create|evaluation.manage|reception.manage)

### Patient evaluation service — `create-visit-consultant` endpoints

Area: warn/block when patients are still checked in there.

- `POST /v1/create-visit-consultant` → VisitController@createVisitConsultant (needs permission:visits.create|visits.update|visits.close|evaluation.manage|reception.manage)

### Patient evaluation service — `create-visit-destination` endpoints

Area: Visit Destinations - Simple endpoint for reception service

- `POST /v1/create-visit-destination` → VisitController@createVisitDestination (needs permission:visits.create|visits.update|visits.close|evaluation.manage|reception.manage)

### Patient evaluation service — `critical-values` endpoints

Area: Additional Lab Settings

- `GET /v1/critical-values/csv-template` → CriticalValueController@csvTemplate
- `POST /v1/critical-values/csv-import` → CriticalValueController@csvImport (needs permission:lab.setup|settings.manage)
- `GET /v1/critical-values` → CriticalValueController@index
- `POST /v1/critical-values` → CriticalValueController@store
- `GET /v1/critical-values/{id}` → CriticalValueController@show
- `PUT/PATCH /v1/critical-values/{id}` → CriticalValueController@update
- `DELETE /v1/critical-values/{id}` → CriticalValueController@destroy

### Patient evaluation service — `cwc-enrollments` endpoints

Area: aren't swallowed by it.

- `GET /v1/cwc-enrollments` → CwcEnrollmentController@index
- `GET /v1/cwc-enrollments/patient/{patientId}` → CwcEnrollmentController@byPatient
- `POST /v1/cwc-enrollments` → CwcEnrollmentController@store (needs permission:clinical_history.record|evaluation.manage)
- `GET /v1/cwc-enrollments/{cwcEnrollment}/schedule-preview` → CwcEnrollmentController@schedulePreview
- `GET /v1/cwc-enrollments/{cwcEnrollment}/schedules` → CwcEnrollmentController@schedules
- `POST /v1/cwc-enrollments/{cwcEnrollment}/confirm` → CwcEnrollmentController@confirm (needs permission:evaluation.manage)
- `PUT /v1/cwc-enrollments/{cwcEnrollment}/schedule` → CwcEnrollmentController@updateSchedule (needs permission:evaluation.manage)
- `POST /v1/cwc-enrollments/{cwcEnrollment}/cancel` → CwcEnrollmentController@cancel (needs permission:evaluation.manage)
- `GET /v1/cwc-enrollments/{cwcEnrollment}` → CwcEnrollmentController@show

### Patient evaluation service — `cwc-visit-schedules` endpoints

Area: aren't swallowed by it.

- `POST /v1/cwc-visit-schedules/{cwcVisitSchedule}/check-in` → CwcVisitScheduleController@checkIn (needs permission:evaluation.manage)

### Patient evaluation service — `dental` endpoints

Area: Dental Extension Module

- `GET /dental/odontogram` → DentalOdontogramController@index
- `GET /dental/odontogram/visit/{visitId}` → DentalOdontogramController@byVisit
- `POST /dental/odontogram` → DentalOdontogramController@store
- `GET /dental/odontogram/{id}` → DentalOdontogramController@show
- `PATCH /dental/odontogram/{id}` → DentalOdontogramController@update
- `DELETE /dental/odontogram/{id}` → DentalOdontogramController@destroy
- `POST /dental/odontogram/{id}/teeth` → DentalOdontogramController@storeTooth
- `PATCH /dental/odontogram/{id}/teeth/{tooth}` → DentalOdontogramController@updateTooth
- `DELETE /dental/odontogram/{id}/teeth/{tooth}` → DentalOdontogramController@destroyTooth
- `POST /dental/odontogram/{id}/surfaces` → DentalOdontogramController@storeSurface
- `PATCH /dental/odontogram/{id}/surfaces/{surface}` → DentalOdontogramController@updateSurface
- `GET /dental/periodontal-chart` → DentalPeriodontalChartController@index
- `GET /dental/periodontal-chart/visit/{visitId}` → DentalPeriodontalChartController@byVisit
- `POST /dental/periodontal-chart` → DentalPeriodontalChartController@store
- `GET /dental/periodontal-chart/{id}` → DentalPeriodontalChartController@show
- `PATCH /dental/periodontal-chart/{id}` → DentalPeriodontalChartController@update
- `DELETE /dental/periodontal-chart/{id}` → DentalPeriodontalChartController@destroy
- `POST /dental/periodontal-chart/{id}/readings` → DentalPeriodontalChartController@storeReading
- `PATCH /dental/periodontal-chart/{id}/readings/{reading}` → DentalPeriodontalChartController@updateReading
- `GET /dental/caries-risk` → DentalCariesRiskAssessmentController@index
- `GET /dental/caries-risk/visit/{visitId}` → DentalCariesRiskAssessmentController@byVisit
- `POST /dental/caries-risk` → DentalCariesRiskAssessmentController@store
- `GET /dental/caries-risk/{id}` → DentalCariesRiskAssessmentController@show
- `PATCH /dental/caries-risk/{id}` → DentalCariesRiskAssessmentController@update
- `DELETE /dental/caries-risk/{id}` → DentalCariesRiskAssessmentController@destroy
- `POST /dental/caries-risk/{id}/calculate-score` → DentalCariesRiskAssessmentController@calculateScore
- `GET /dental/claims` → DentalClaimController@index
- `GET /dental/claims/visit/{visitId}` → DentalClaimController@byVisit
- `POST /dental/claims` → DentalClaimController@store
- `GET /dental/claims/{id}` → DentalClaimController@show
- `PATCH /dental/claims/{id}` → DentalClaimController@update
- `DELETE /dental/claims/{id}` → DentalClaimController@destroy
- `POST /dental/claims/{id}/items` → DentalClaimController@storeItem
- `PATCH /dental/claims/{id}/items/{item}` → DentalClaimController@updateItem
- `DELETE /dental/claims/{id}/items/{item}` → DentalClaimController@destroyItem
- `POST /dental/claims/{id}/submit` → DentalClaimController@submit
- `GET /dental/insurance-breakdowns` → DentalInsuranceBreakdownController@index
- `GET /dental/insurance-breakdowns/{id}` → DentalInsuranceBreakdownController@show
- `POST /dental/insurance-breakdowns/{claim}/calculate` → DentalInsuranceBreakdownController@calculate
- `GET /dental/clearance` → DentalClearanceController@index
- `GET /dental/clearance/visit/{visitId}` → DentalClearanceController@byVisit
- `POST /dental/clearance` → DentalClearanceController@store
- `GET /dental/clearance/{id}` → DentalClearanceController@show
- `PATCH /dental/clearance/{id}` → DentalClearanceController@update
- `DELETE /dental/clearance/{id}` → DentalClearanceController@destroy
- `POST /dental/clearance/{id}/sign` → DentalClearanceController@sign
- `POST /dental/clearance/{id}/amend` → DentalClearanceController@amend

### Patient evaluation service — `destinations` endpoints

Area: warn/block when patients are still checked in there.

- `GET /v1/destinations/{destinationId}/active-count` → VisitController@activeDestinationCount (needs permission:visits.create|visits.update|visits.close|evaluation.manage|reception.manage)

### Patient evaluation service — `diagnoses` endpoints

Area: was actually diagnosed.

- `GET /v1/diagnoses` → DiagnosisController@index

### Patient evaluation service — `diagnosis-codes` endpoints

Area: Diagnosis Codes (ICD-10 and MOH 705); token, so core authenticates it like every other SHA call.

- `GET /v1/diagnosis-codes/icd10` → DiagnosisCodeController@icd10
- `GET /v1/diagnosis-codes/moh705` → DiagnosisCodeController@moh705
- `GET /v1/diagnosis-codes/icd11` → DiagnosisCodeController@icd11
- `POST /v1/diagnosis-codes/icd11/sync` → DiagnosisCodeController@syncIcd11 (needs permission:settings.manage)
- `GET /v1/diagnosis-codes` → DiagnosisCodeController@index
- `POST /v1/diagnosis-codes` → DiagnosisCodeController@store
- `GET /v1/diagnosis-codes/{id}` → DiagnosisCodeController@show
- `PUT/PATCH /v1/diagnosis-codes/{id}` → DiagnosisCodeController@update
- `DELETE /v1/diagnosis-codes/{id}` → DiagnosisCodeController@destroy

### Patient evaluation service — `doctor-notes` endpoints

Area: Additional Doctor Notes endpoints; Doctor Notes Management

- `GET /v1/doctor-notes/follow-ups` → DoctorNoteController@followUps
- `POST /v1/doctor-notes/{doctorNote}/sign` → DoctorNoteController@sign (needs permission:clinical_notes.sign|evaluations.sign|evaluation.manage)
- `POST /v1/doctor-notes/{doctorNote}/finalize` → DoctorNoteController@finalize (needs permission:evaluations.finalize|clinical_notes.sign|evaluation.manage)
- `POST /v1/doctor-notes/{doctorNote}/mark-next-steps-done` → DoctorNoteController@markNextStepsDone (needs permission:clinical_notes.update|evaluation.manage)
- `GET /v1/doctor-notes` → DoctorNoteController@byVisitAndUser
- `GET /v1/doctor-notes` → DoctorNoteController@index
- `POST /v1/doctor-notes` → DoctorNoteController@store
- `GET /v1/doctor-notes/{id}` → DoctorNoteController@show
- `PUT/PATCH /v1/doctor-notes/{id}` → DoctorNoteController@update
- `DELETE /v1/doctor-notes/{id}` → DoctorNoteController@destroy
- `GET /v1/doctor-notes/previous/{patientId}` → DoctorNoteController@previousNotes

### Patient evaluation service — `document-types` endpoints

Area: Document Types

- `GET /v1/document-types` → DocumentTypeController@index
- `POST /v1/document-types` → DocumentTypeController@store
- `GET /v1/document-types/{id}` → DocumentTypeController@show
- `PUT/PATCH /v1/document-types/{id}` → DocumentTypeController@update
- `DELETE /v1/document-types/{id}` → DocumentTypeController@destroy

### Patient evaluation service — `drug-label-templates` endpoints

Area: Drug label instruction templates — one per drug per organization

- `GET /v1/drug-label-templates/{drugId}` → DrugLabelTemplateController@show
- `PUT /v1/drug-label-templates/{drugId}` → DrugLabelTemplateController@upsert

### Patient evaluation service — `eabl-wheels` endpoints

Area: table never existed in v3 before.

- `GET /v1/eabl-wheels` → EablAssessmentWheelController@index

### Patient evaluation service — `ent` endpoints

Area: ENT (Otolaryngology) Extension Module

- `GET /ent/intake` → EntIntakeController@index
- `GET /ent/intake/visit/{visitId}` → EntIntakeController@byVisit
- `POST /ent/intake` → EntIntakeController@store
- `GET /ent/intake/{id}` → EntIntakeController@show
- `PATCH /ent/intake/{id}` → EntIntakeController@update
- `DELETE /ent/intake/{id}` → EntIntakeController@destroy
- `GET /ent/symptom-map/visit/{visitId}` → EntSymptomMapController@byVisit
- `POST /ent/symptom-map` → EntSymptomMapController@store
- `DELETE /ent/symptom-map/points/{id}` → EntSymptomMapController@destroyPoint
- `GET /ent/ear-exam/visit/{visitId}` → EntEarExamController@byVisit
- `POST /ent/ear-exam` → EntEarExamController@store
- `GET /ent/head-neck-exam/visit/{visitId}` → EntHeadNeckExamController@byVisit
- `POST /ent/head-neck-exam` → EntHeadNeckExamController@store
- `GET /ent/audiograms/visit/{visitId}` → EntAudiogramController@byVisit
- `POST /ent/audiograms` → EntAudiogramController@store
- `GET /ent/audiograms/{id}` → EntAudiogramController@show
- `GET /ent/audiograms/patient/{patientId}/history` → EntAudiogramController@history
- `GET /ent/audiograms/{id}/compare` → EntAudiogramController@compare
- `GET /ent/tympanometry/visit/{visitId}` → EntTympanometryController@byVisit
- `POST /ent/tympanometry` → EntTympanometryController@store
- `GET /ent/tympanometry/{id}` → EntTympanometryController@show
- `GET /ent/vestibular/test-definitions` → EntVestibularController@testDefinitions
- `GET /ent/vestibular/visit/{visitId}` → EntVestibularController@byVisit
- `POST /ent/vestibular` → EntVestibularController@store
- `GET /ent/tinnitus-assessment/visit/{visitId}` → EntTinnitusController@byVisit
- `POST /ent/tinnitus-assessment` → EntTinnitusController@store
- `GET /ent/nasal-sinus/visit/{visitId}` → EntNasalSinusController@byVisit
- `POST /ent/nasal-sinus` → EntNasalSinusController@store
- `GET /ent/sleep-airway/visit/{visitId}` → EntSleepAirwayController@byVisit
- `POST /ent/sleep-airway` → EntSleepAirwayController@store
- `GET /ent/voice-assessment/visit/{visitId}` → EntVoiceController@byVisit
- `POST /ent/voice-assessment` → EntVoiceController@store
- `GET /ent/swallowing-assessment/visit/{visitId}` → EntSwallowingController@byVisit
- `POST /ent/swallowing-assessment` → EntSwallowingController@store
- `GET /ent/reflux-assessment/visit/{visitId}` → EntRefluxController@byVisit
- `POST /ent/reflux-assessment` → EntRefluxController@store
- `GET /ent/endoscopy/visit/{visitId}` → EntEndoscopyController@byVisit
- `POST /ent/endoscopy` → EntEndoscopyController@store
- `GET /ent/endoscopy/{id}` → EntEndoscopyController@show
- `PATCH /ent/endoscopy/{id}` → EntEndoscopyController@update
- `GET /ent/preop` → EntPreopController@index
- `GET /ent/preop/visit/{visitId}` → EntPreopController@byVisit
- `POST /ent/preop` → EntPreopController@store
- `GET /ent/preop/{id}` → EntPreopController@show
- `PATCH /ent/preop/{id}` → EntPreopController@update
- `DELETE /ent/preop/{id}` → EntPreopController@destroy
- `GET /ent/consent` → EntConsentController@index
- `GET /ent/consent/visit/{visitId}` → EntConsentController@byVisit
- `POST /ent/consent` → EntConsentController@store
- `GET /ent/consent/{id}` → EntConsentController@show
- `PATCH /ent/consent/{id}` → EntConsentController@update
- `DELETE /ent/consent/{id}` → EntConsentController@destroy
- `POST /ent/consent/{id}/sign` → EntConsentController@sign
- `POST /ent/consent/{id}/amend` → EntConsentController@amend
- `GET /ent/procedure-record/visit/{visitId}` → EntProcedureRecordController@byVisit
- `POST /ent/procedure-record` → EntProcedureRecordController@store
- `GET /ent/procedure-record/{id}` → EntProcedureRecordController@show
- `PATCH /ent/procedure-record/{id}` → EntProcedureRecordController@update
- `DELETE /ent/procedure-record/{id}` → EntProcedureRecordController@destroy
- `GET /ent/device-implant/visit/{visitId}` → EntDeviceImplantController@byVisit
- `POST /ent/device-implant` → EntDeviceImplantController@store
- `GET /ent/device-implant/{id}` → EntDeviceImplantController@show
- `PATCH /ent/device-implant/{id}` → EntDeviceImplantController@update
- `DELETE /ent/device-implant/{id}` → EntDeviceImplantController@destroy
- `GET /ent/functional-instruments` → EntFunctionalInstrumentController@index
- `GET /ent/functional-instruments/active` → EntFunctionalInstrumentController@active
- `GET /ent/functional-instruments/{id}` → EntFunctionalInstrumentController@show
- `GET /ent/functional-assessments/visit/{visitId}` → EntFunctionalAssessmentController@byVisit
- `GET /ent/functional-assessments/patient/{patientId}/history` → EntFunctionalAssessmentController@history
- `POST /ent/functional-assessments` → EntFunctionalAssessmentController@store
- `GET /ent/functional-assessments/{id}` → EntFunctionalAssessmentController@show

### Patient evaluation service — `examination-review-details` endpoints

Area: Documentation & Drawing images on the ophthalmologist notes screen

- `GET /examination-review-details` → ExaminationReviewDetailsController@index
- `POST /examination-review-details` → ExaminationReviewDetailsController@store
- `GET /examination-review-details/{id}` → ExaminationReviewDetailsController@show
- `PUT/PATCH /examination-review-details/{id}` → ExaminationReviewDetailsController@update
- `DELETE /examination-review-details/{id}` → ExaminationReviewDetailsController@destroy

### Patient evaluation service — `examination-reviews` endpoints

Area: Documentation & Drawing images on the ophthalmologist notes screen

- `GET /examination-reviews` → ExaminationReviewController@index
- `POST /examination-reviews` → ExaminationReviewController@store
- `GET /examination-reviews/{id}` → ExaminationReviewController@show
- `PUT/PATCH /examination-reviews/{id}` → ExaminationReviewController@update
- `DELETE /examination-reviews/{id}` → ExaminationReviewController@destroy

### Patient evaluation service — `follow-up` endpoints

Area: ENT (Otolaryngology) Extension Module

- `GET /follow-up` → FollowUpController@index
- `POST /follow-up` → FollowUpController@store (needs permission:clinical_notes.create|evaluation.manage)

### Patient evaluation service — `follow-ups` endpoints

Area: ENT (Otolaryngology) Extension Module

- `GET /follow-ups` → FollowUpController@index
- `POST /follow-ups` → FollowUpController@store
- `GET /follow-ups/{id}` → FollowUpController@show
- `PUT/PATCH /follow-ups/{id}` → FollowUpController@update
- `DELETE /follow-ups/{id}` → FollowUpController@destroy

### Patient evaluation service — `formulae` endpoints

Area: Additional Lab Settings

- `GET /v1/formulae` → FormulaController@index
- `POST /v1/formulae` → FormulaController@store
- `GET /v1/formulae/{id}` → FormulaController@show
- `PUT/PATCH /v1/formulae/{id}` → FormulaController@update
- `DELETE /v1/formulae/{id}` → FormulaController@destroy

### Patient evaluation service — `fp-services` endpoints

Area: as an {fpService} id.; before the '{ancEnrollment}' wildcard so they aren't swallowed by it.

- `GET /v1/fp-services` → FpServiceController@index
- `GET /v1/fp-services/summary` → FpServiceController@summary
- `GET /v1/fp-services/patient/{patientId}` → FpServiceController@byPatient
- `POST /v1/fp-services` → FpServiceController@store (needs permission:clinical_history.record|evaluation.manage)
- `PUT /v1/fp-services/{fpService}` → FpServiceController@update (needs permission:clinical_history.record|evaluation.manage)
- `DELETE /v1/fp-services/{fpService}` → FpServiceController@destroy (needs permission:evaluation.manage)
- `GET /v1/fp-services/{fpService}` → FpServiceController@show

### Patient evaluation service — `gateway` endpoints

Area: Model Gateway — cross-tenant data access and replication

- `POST /gateway` → ModelGatewayController@handle

### Patient evaluation service — `health` endpoints

Area: Health check endpoint (unversioned, public)

- `GET /health` → HealthController@index

### Patient evaluation service — `hie-submissions` endpoints

Area: §6 Phase 3.

- `GET /v2/hie-submissions/{patientId}` → HieSubmissionController@show

### Patient evaluation service — `history` endpoints

Area: show route swallows this as visit="dashboard-summary".

- `GET /v1/history/outpatient-notes/logs/{visitId}` → PatientHistoryController@fetchOutPatientNotesViews
- `POST /v1/history/remove-item` → PatientHistoryController@removeItem (needs permission:evaluation.manage)

### Patient evaluation service — `icd10` endpoints

Area: Diagnosis Codes (ICD-10 and MOH 705)

- `GET /v1/icd10/note` → VisitIcd10VariationController@index

### Patient evaluation service — `internal` endpoints

Area: console job (no user token to send) — see VerifyInternalSignedRequest.; no user token, so it settles a paid POS sale's lines the signed way.; of the same resource name registered further down this file.

- `GET /v1/internal/visits/open` → VisitController@internalOpenVisits
- `POST /v1/internal/visits/{visitId}/auto-close` → VisitController@internalAutoClose
- `GET /v1/internal/procedure-categories` → ProcedureCategoryController@internalIndex
- `POST /v1/internal/procedure-categories/seed-defaults` → ProcedureCategoryController@internalSeedDefaults
- `POST /v1/internal/procedure-categories/csv-import` → ProcedureCategoryController@internalCsvImport
- `POST /v1/internal/procedures/seed-defaults` → ProcedureController@internalSeedDefaults
- `POST /v1/internal/procedures/csv-import` → ProcedureController@internalCsvImport
- `POST /v1/internal/organizations/{id}/seed-evaluation-catalogue` → ProcedureController@internalSeedFullCatalogue
- `POST /v1/internal/prescriptions/pos-paid` → PrescriptionController@internalPosPaid

### Patient evaluation service — `investigation-results` endpoints

Area: Investigation Result Reviews; Investigation Results

- `GET /v1/investigation-results` → InvestigationResultController@index
- `POST /v1/investigation-results/store` → InvestigationResultController@storeResult (needs permission:investigations.record_results|lab.results.record|lab.results.update|evaluation.manage)
- `PATCH /v1/investigation-results/update-status` → InvestigationResultController@updateStatus (needs permission:investigations.record_results|lab.results.record|lab.results.update|evaluation.manage)
- `PATCH /v1/investigation-results/update-receive-status` → InvestigationResultController@updateReceiveStatus (needs permission:lab.samples.receive|investigations.record_results|lab.results.record|evaluation.manage)
- `PATCH /v1/investigation-results/update-result` → InvestigationResultController@updateResult (needs permission:investigations.record_results|lab.results.record|lab.results.update|evaluation.manage)
- `POST /v1/investigation-results/email` → InvestigationResultController@emailResults (needs permission:investigations.release_results|lab.results.release|evaluation.manage)
- `POST /v1/investigation-results/submit-review` → InvestigationResultController@submitReview (needs permission:investigations.verify_results|lab.results.verify|evaluation.manage)
- `GET /v1/investigation-results/{id}/reviewed` → InvestigationResultController@fetchReviewedInvestigation
- `PATCH /v1/investigation-results/{id}/result` → InvestigationResultController@updateResultValue (needs permission:investigations.record_results|lab.results.record|lab.results.update|evaluation.manage)
- `GET /v1/investigation-results/analytics/fetch` → InvestigationResultController@fetchResultsAnalytics
- `GET /v1/investigation-results/analytics/print` → InvestigationResultController@printAnalytics
- `POST /v1/investigation-results/groups` → InvestigationResultController@saveGroup (needs permission:lab.setup|settings.manage|lab.results.record)
- `GET /v1/investigation-results/groups` → InvestigationResultController@groups
- `GET /v1/investigation-results/review` → InvestigationResultReviewController@index
- `POST /v1/investigation-results/review` → InvestigationResultReviewController@store
- `GET /v1/investigation-results/review/{id}` → InvestigationResultReviewController@show
- `PUT/PATCH /v1/investigation-results/review/{id}` → InvestigationResultReviewController@update
- `DELETE /v1/investigation-results/review/{id}` → InvestigationResultReviewController@destroy

### Patient evaluation service — `investigations` endpoints

Area: below — that wildcard would otherwise swallow this literal path.; caller can't touch anything clinical through it.; invoiced, paid, resulted or cancelled.

- `GET /v1/investigations/staff-activity` → InvestigationController@staffActivity
- `GET /v1/investigations/radiology-request-form` → InvestigationController@radiologyRequestForm
- `POST /v1/investigations/{investigation}/collect-sample` → InvestigationController@collectSample (needs permission:lab.samples.collect|investigations.request|evaluation.manage)
- `POST /v1/investigations/{investigation}/complete` → InvestigationController@complete (needs permission:investigations.record_results|lab.results.record|evaluation.manage)
- `POST /v1/investigations/{investigation}/verify` → InvestigationController@verify (needs permission:investigations.verify_results|lab.results.verify|evaluation.manage)
- `POST /v1/investigations/{investigation}/cancel` → InvestigationController@cancel (needs permission:investigations.cancel|lab.tests.cancel|evaluation.manage)
- `PUT /v1/investigations/{investigation}/billing-sync` → InvestigationController@markBillingSync (needs permission:finance.process_payments|billing.manage|investigations.request|investigations.record_results|evaluation.manage)
- `GET /v1/investigations` → InvestigationController@index
- `POST /v1/investigations` → InvestigationController@store
- `GET /v1/investigations/{id}` → InvestigationController@show
- `PUT/PATCH /v1/investigations/{id}` → InvestigationController@update
- `DELETE /v1/investigations/{id}` → InvestigationController@destroy

### Patient evaluation service — `lab-test-additives` endpoints

Area: Lab Settings CRUD

- `GET /v1/lab-test-additives` → LabTestAdditiveController@index
- `POST /v1/lab-test-additives` → LabTestAdditiveController@store
- `GET /v1/lab-test-additives/{id}` → LabTestAdditiveController@show
- `PUT/PATCH /v1/lab-test-additives/{id}` → LabTestAdditiveController@update
- `DELETE /v1/lab-test-additives/{id}` → LabTestAdditiveController@destroy

### Patient evaluation service — `lab-test-categories` endpoints

Area: Lab Settings CRUD

- `GET /v1/lab-test-categories` → LabTestCategoryController@index
- `POST /v1/lab-test-categories` → LabTestCategoryController@store
- `GET /v1/lab-test-categories/{id}` → LabTestCategoryController@show
- `PUT/PATCH /v1/lab-test-categories/{id}` → LabTestCategoryController@update
- `DELETE /v1/lab-test-categories/{id}` → LabTestCategoryController@destroy

### Patient evaluation service — `laboratory-requests` endpoints

Area: Walk-in laboratory requests (no visit required)

- `POST /v1/laboratory-requests` → LaboratoryRequestController@store (needs permission:investigations.request|lab.tests.request|evaluation.manage|reception.manage)

### Patient evaluation service — `machines` endpoints

Area: Additional Lab Settings

- `GET /v1/machines` → MachineController@index
- `POST /v1/machines` → MachineController@store
- `GET /v1/machines/{id}` → MachineController@show
- `PUT/PATCH /v1/machines/{id}` → MachineController@update
- `DELETE /v1/machines/{id}` → MachineController@destroy
- `GET /v1/machines/{machine}/code-mappings` → MachineCodeMappingController@index
- `POST /v1/machines/{machine}/code-mappings` → MachineCodeMappingController@store (needs permission:lab.setup|settings.manage)
- `DELETE /v1/machines/{machine}/code-mappings/{id}` → MachineCodeMappingController@destroy (needs permission:lab.setup|settings.manage)

### Patient evaluation service — `medication-requests` endpoints

Area: brand-new routes with no existing caller to preserve compatibility for.; §6 Phase 3.

- `GET /v2/medication-requests` → MedicationRequestController@index
- `GET /v2/medication-requests/{id}` → MedicationRequestController@show
- `POST /v2/medication-requests` → MedicationRequestController@store

### Patient evaluation service — `moh-variations` endpoints

Area: Frontend nests the actual fields under `slit_lamp_exam`. Unwrap if present.

- `GET /moh-variations` → MohVariationsController@index
- `POST /moh-variations` → MohVariationsController@store
- `GET /moh-variations/{id}` → MohVariationsController@show
- `PUT/PATCH /moh-variations/{id}` → MohVariationsController@update
- `DELETE /moh-variations/{id}` → MohVariationsController@destroy

### Patient evaluation service — `observations` endpoints

Area: brand-new routes with no existing caller to preserve compatibility for.; §6 Phase 3.

- `GET /v2/observations` → ObservationController@index
- `GET /v2/observations/{id}` → ObservationController@show
- `POST /v2/observations` → ObservationController@store

### Patient evaluation service — `obstetric-history` endpoints

Area: so they ride the same three routes.

- `GET /v1/obstetric-history/{visit}` → ObstetricHistoryController@index
- `POST /v1/obstetric-history` → ObstetricHistoryController@store
- `PUT /v1/obstetric-history/{obstetricHistory}` → ObstetricHistoryController@update
- `DELETE /v1/obstetric-history/{obstetricHistory}` → ObstetricHistoryController@destroy

### Patient evaluation service — `oncology` endpoints

Area: (§6.4); the policy checks the verifier as well as the caller.; A human marks a referral lost; the nightly job never does (§8.5).; Approval is held by the oncologist role only (§5.1).; Safe to call repeatedly; it never changes cycle state.; The ONLY way to change a diagnosis — there is no PUT.; The SAE governance queue — clinical lead and governance roles only.

- `GET /oncology/screenings/eligibility` → OncologyScreeningController@eligibility (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/screenings/due` → OncologyScreeningController@due (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/screenings/summary` → OncologyScreeningController@summary (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/screenings` → OncologyScreeningController@index (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/screenings` → OncologyScreeningController@store (needs permission:oncology.screening.record|oncology.manage)
- `GET /oncology/screenings/{screening}` → OncologyScreeningController@show (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/referrals` → OncologyReferralController@index (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/referrals/tracing-queue` → OncologyReferralController@tracingQueue (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/referrals/{referral}` → OncologyReferralController@show (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/referrals/{referral}/outcome` → OncologyReferralController@outcome (needs permission:oncology.diagnosis.write|oncology.screening.record|oncology.manage)
- `POST /oncology/referrals/{referral}/attempts` → OncologyReferralController@attempts (needs permission:oncology.screening.record|oncology.manage)
- `POST /oncology/referrals/{referral}/mark-lost` → OncologyReferralController@markLost (needs permission:oncology.screening.record|oncology.manage)
- `GET /oncology/diagnoses` → OncologyDiagnosisController@index (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/diagnoses` → OncologyDiagnosisController@store (needs permission:oncology.diagnosis.write|oncology.manage)
- `GET /oncology/diagnoses/{diagnosis}` → OncologyDiagnosisController@show (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/diagnoses/{diagnosis}/revisions` → OncologyDiagnosisController@revise (needs permission:oncology.diagnosis.write|oncology.manage)
- `POST /oncology/diagnoses/{diagnosis}/stagings` → OncologyDiagnosisController@storeStaging (needs permission:oncology.staging.write|oncology.manage)
- `POST /oncology/diagnoses/{diagnosis}/biomarkers` → OncologyDiagnosisController@storeBiomarker (needs permission:oncology.diagnosis.write|oncology.manage)
- `GET /oncology/mdt/meetings` → OncologyMdtController@index (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/mdt/meetings` → OncologyMdtController@store (needs permission:oncology.mdt.decide|oncology.manage)
- `GET /oncology/mdt/meetings/{meeting}` → OncologyMdtController@show (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/mdt/meetings/{meeting}/cases` → OncologyMdtController@storeCase (needs permission:oncology.mdt.decide|oncology.manage)
- `POST /oncology/mdt/cases/{case}/decision` → OncologyMdtController@decide (needs permission:oncology.mdt.decide|oncology.manage)
- `GET /oncology/treatment-plans` → OncologyTreatmentPlanController@index (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/treatment-plans` → OncologyTreatmentPlanController@store (needs permission:oncology.plan.approve|oncology.diagnosis.write|oncology.manage)
- `GET /oncology/treatment-plans/{plan}` → OncologyTreatmentPlanController@show (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/treatment-plans/{plan}/projection` → OncologyTreatmentPlanController@projection (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/treatment-plans/{plan}/approve` → OncologyTreatmentPlanController@approve (needs permission:oncology.plan.approve)
- `POST /oncology/treatment-plans/{plan}/close` → OncologyTreatmentPlanController@close (needs permission:oncology.plan.approve|oncology.manage)
- `GET /oncology/treatment-plans/{plan}/toxicity-grid` → OncologyToxicityController@grid (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/worklists/day-unit` → OncologyCycleController@dayUnit (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/cycles/{cycle}` → OncologyCycleController@show (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/cycles/{cycle}/gate` → OncologyCycleController@gate (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/cycles/{cycle}/dose-sheet` → OncologyCycleController@doseSheet (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/cycles/{cycle}/open` → OncologyCycleController@open (needs permission:oncology.cycle.open|oncology.manage)
- `POST /oncology/cycles/{cycle}/defer` → OncologyCycleController@defer (needs permission:oncology.cycle.open|oncology.manage)
- `POST /oncology/cycles/{cycle}/skip` → OncologyCycleController@skip (needs permission:oncology.plan.approve|oncology.manage)
- `POST /oncology/cycles/{cycle}/administrations` → OncologyCycleController@storeAdministration (needs permission:oncology.cytotoxic.handle|oncology.manage)
- `POST /oncology/administrations/{administration}/revisions` → OncologyCycleController@reviseAdministration (needs permission:oncology.cytotoxic.handle|oncology.manage)
- `GET /oncology/ctcae-terms` → OncologyToxicityController@terms (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/toxicities` → OncologyToxicityController@index (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/toxicities` → OncologyToxicityController@store (needs permission:oncology.toxicity.write|clinical_history.record|evaluation.manage)
- `GET /oncology/toxicities/serious` → OncologyToxicityController@seriousQueue (needs permission:oncology.sae.view)
- `POST /oncology/toxicities/{toxicity}/reported` → OncologyToxicityController@markSaeReported (needs permission:oncology.sae.view)
- `GET /oncology/response-assessments` → OncologyResponseController@index (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/response-assessments/rate` → OncologyResponseController@rate (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/response-assessments` → OncologyResponseController@store (needs permission:oncology.toxicity.write|oncology.manage)
- `GET /oncology/benefit/tariffs` → OncologyBenefitController@tariffs (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/benefit/lines` → OncologyBenefitController@lines (needs permission:oncology.benefit.view|oncology.manage)
- `POST /oncology/benefit/periods` → OncologyBenefitController@storePeriod (needs permission:oncology.benefit.view|oncology.manage)
- `POST /oncology/benefit/periods/{period}/external` → OncologyBenefitController@syncExternal (needs permission:oncology.benefit.view|oncology.manage)
- `GET /oncology/benefit/{patient}` → OncologyBenefitController@show (needs permission:oncology.benefit.view|oncology.manage)
- `GET /oncology/care-plan-items` → OncologyPalliativeController@items (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/care-plan-items` → OncologyPalliativeController@storeItem (needs permission:oncology.toxicity.write|oncology.manage)
- `POST /oncology/care-plan-items/{item}/attend` → OncologyPalliativeController@attendItem (needs permission:oncology.toxicity.write|oncology.manage)
- `POST /oncology/care-plan-items/{item}/attempts` → OncologyPalliativeController@storeAttempt (needs permission:oncology.screening.record|oncology.manage)
- `GET /oncology/palliative/assessments` → OncologyPalliativeController@assessments (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/palliative/assessments` → OncologyPalliativeController@storeAssessment (needs permission:oncology.toxicity.write|clinical_history.record|evaluation.manage)
- `GET /oncology/palliative/controlled-drug-register` → OncologyPalliativeController@controlledDrugRegister (needs permission:oncology.benefit.view|pharmacy.dispensing.view|oncology.manage)
- `GET /oncology/tracing-queue` → OncologyPalliativeController@tracingQueue (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/protocols` → OncologyRegistryController@protocols (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/protocols/{protocol}` → OncologyRegistryController@protocol (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/screening-programmes` → OncologyRegistryController@screeningProgrammes (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/surveillance-schedules` → OncologyRegistryController@surveillanceSchedules (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/parameters` → OncologyRegistryController@parameters (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `GET /oncology/observations` → OncologyRegistryController@observations (needs permission:oncology.view|oncology.manage|evaluation.manage)
- `POST /oncology/observations` → OncologyRegistryController@storeObservation (needs permission:oncology.toxicity.write|clinical_history.record|evaluation.manage)
- `GET /oncology/patients/{patient}/summary` → OncologySummaryController@show (needs permission:oncology.view|oncology.manage|evaluation.manage)

### Patient evaluation service — `operating-store` endpoints

Area: Operating Store Selection (session-scoped per authenticated login token)

- `GET /v1/operating-store/session` → OperatingStoreSessionController@show
- `POST /v1/operating-store/session` → OperatingStoreSessionController@store
- `DELETE /v1/operating-store/session` → OperatingStoreSessionController@destroy

### Patient evaluation service — `ophthalmology` endpoints

Area: Eyewear prescription (incl. pupillary distance) — stored as JSON on evaluation_refraction.eyewear_prescription; Frontend nests the actual fields under `slit_lamp_exam`. Unwrap if present.; Same row the refraction index/store use: the visit's newest.; Slit lamp exam

- `GET /ophthalmology/eyewear-pres`
- `POST /ophthalmology/eyewear-pres`
- `GET /ophthalmology/slit-lamp-exam`
- `POST /ophthalmology/slit-lamp-exam`

### Patient evaluation service — `ophthalmology-images` endpoints

Area: Documentation & Drawing images on the ophthalmologist notes screen

- `GET /ophthalmology-images` → OphthalmologyImageController@index
- `POST /ophthalmology-images` → OphthalmologyImageController@store (needs permission:refraction.record|evaluation.manage)
- `GET /ophthalmology-images/{image}/view` → OphthalmologyImageController@view
- `DELETE /ophthalmology-images/{image}` → OphthalmologyImageController@destroy (needs permission:refraction.record|evaluation.manage)

### Patient evaluation service — `ophthalmology-notes` endpoints

Area: Frontend nests the actual fields under `slit_lamp_exam`. Unwrap if present.

- `GET /ophthalmology-notes` → OphthalmologyNotesController@index
- `POST /ophthalmology-notes` → OphthalmologyNotesController@store
- `GET /ophthalmology-notes/{id}` → OphthalmologyNotesController@show
- `PUT/PATCH /ophthalmology-notes/{id}` → OphthalmologyNotesController@update
- `DELETE /ophthalmology-notes/{id}` → OphthalmologyNotesController@destroy

### Patient evaluation service — `orthopedics` endpoints

Area: Orthopedic Extension Module

- `GET /orthopedics/intake` → OrthopedicIntakeController@index
- `GET /orthopedics/intake/visit/{visitId}` → OrthopedicIntakeController@byVisit
- `POST /orthopedics/intake` → OrthopedicIntakeController@store
- `GET /orthopedics/intake/{id}` → OrthopedicIntakeController@show
- `PATCH /orthopedics/intake/{id}` → OrthopedicIntakeController@update
- `DELETE /orthopedics/intake/{id}` → OrthopedicIntakeController@destroy
- `GET /orthopedics/injury-pain/visit/{visitId}` → OrthopedicInjuryPainController@byVisit
- `POST /orthopedics/injury-pain` → OrthopedicInjuryPainController@store
- `DELETE /orthopedics/injury-pain/pain-locations/{id}` → OrthopedicInjuryPainController@destroyPainLocation
- `GET /orthopedics/msk-exam/test-definitions` → OrthopedicMskExamController@testDefinitions
- `GET /orthopedics/msk-exam/visit/{visitId}` → OrthopedicMskExamController@byVisit
- `POST /orthopedics/msk-exam` → OrthopedicMskExamController@store
- `GET /orthopedics/rom-strength/visit/{visitId}` → OrthopedicRomStrengthController@byVisit
- `POST /orthopedics/rom-strength` → OrthopedicRomStrengthController@store
- `GET /orthopedics/gait-posture/visit/{visitId}` → OrthopedicGaitPostureController@byVisit
- `POST /orthopedics/gait-posture` → OrthopedicGaitPostureController@store
- `GET /orthopedics/preop` → OrthopedicPreopController@index
- `GET /orthopedics/preop/visit/{visitId}` → OrthopedicPreopController@byVisit
- `POST /orthopedics/preop` → OrthopedicPreopController@store
- `GET /orthopedics/preop/{id}` → OrthopedicPreopController@show
- `PATCH /orthopedics/preop/{id}` → OrthopedicPreopController@update
- `DELETE /orthopedics/preop/{id}` → OrthopedicPreopController@destroy
- `GET /orthopedics/consent` → OrthopedicConsentController@index
- `GET /orthopedics/consent/visit/{visitId}` → OrthopedicConsentController@byVisit
- `POST /orthopedics/consent` → OrthopedicConsentController@store
- `GET /orthopedics/consent/{id}` → OrthopedicConsentController@show
- `PATCH /orthopedics/consent/{id}` → OrthopedicConsentController@update
- `DELETE /orthopedics/consent/{id}` → OrthopedicConsentController@destroy
- `POST /orthopedics/consent/{id}/sign` → OrthopedicConsentController@sign
- `POST /orthopedics/consent/{id}/amend` → OrthopedicConsentController@amend
- `GET /orthopedics/implants/visit/{visitId}` → OrthopedicImplantController@byVisit
- `POST /orthopedics/implants` → OrthopedicImplantController@store
- `GET /orthopedics/implants/{id}` → OrthopedicImplantController@show
- `PATCH /orthopedics/implants/{id}` → OrthopedicImplantController@update
- `DELETE /orthopedics/implants/{id}` → OrthopedicImplantController@destroy
- `GET /orthopedics/functional-instruments` → OrthopedicFunctionalInstrumentController@index
- `GET /orthopedics/functional-instruments/active` → OrthopedicFunctionalInstrumentController@active
- `GET /orthopedics/functional-instruments/{id}` → OrthopedicFunctionalInstrumentController@show
- `GET /orthopedics/functional-assessments/visit/{visitId}` → OrthopedicFunctionalAssessmentController@byVisit
- `GET /orthopedics/functional-assessments/patient/{patientId}/history` → OrthopedicFunctionalAssessmentController@history
- `POST /orthopedics/functional-assessments` → OrthopedicFunctionalAssessmentController@store
- `GET /orthopedics/functional-assessments/{id}` → OrthopedicFunctionalAssessmentController@show

### Patient evaluation service — `patient-documents` endpoints

Area: Patient Documents

- `GET /v1/patient-documents/{document}/view` → PatientDocumentController@view

### Patient evaluation service — `patient-history` endpoints

Area: show route swallows this as visit="dashboard-summary".

- `GET /v1/patient-history/{patientId}` → PatientHistoryController@index (needs permission:patients.view_history|evaluations.view|evaluation.manage)
- `GET /v1/patient-history/{patientId}/vitals` → PatientHistoryController@vitals (needs permission:patients.view_history|evaluations.view|evaluation.manage|vitals.view)
- `GET /v1/patient-history/{patientId}/visits` → PatientHistoryController@visits (needs permission:patients.view_history|evaluations.view|evaluation.manage)
- `GET /v1/patient-history/{patientId}/fhir` → PatientHistoryController@fhir (needs permission:patients.view_history|evaluations.view|evaluation.manage)

### Patient evaluation service — `patient-meta` endpoints

Area: Patient Meta

- `GET /v1/patient-meta/{patientId}` → PatientMetaController@show

### Patient evaluation service — `patients` endpoints

Area: Aggregate all investigations for a patient (all visits); Aggregate all prescriptions for a patient (all visits); Aggregate all visits for a patient (all visits); Patient-specific visits endpoint (overrides default index for /patients/{patient}/visits)

- `GET /v1/patients/{patient}/all-investigations` → PatientEvaluationProxyController@investigations
- `GET /v1/patients/{patient}/all-prescriptions` → PatientEvaluationPrescriptionProxyController@prescriptions
- `GET /v1/patients/{patient}/all-visits` → PatientEvaluationVisitProxyController@visits
- `GET /v1/patients/{patient}/visits` → VisitController@patientVisits

### Patient evaluation service — `pnc-enrollments` endpoints

Area: wildcard so they aren't swallowed by it.

- `GET /v1/pnc-enrollments` → PncEnrollmentController@index
- `GET /v1/pnc-enrollments/patient/{patientId}` → PncEnrollmentController@byPatient
- `POST /v1/pnc-enrollments` → PncEnrollmentController@store (needs permission:clinical_history.record|evaluation.manage)
- `GET /v1/pnc-enrollments/{pncEnrollment}/schedule-preview` → PncEnrollmentController@schedulePreview
- `GET /v1/pnc-enrollments/{pncEnrollment}/schedules` → PncEnrollmentController@schedules
- `POST /v1/pnc-enrollments/{pncEnrollment}/confirm` → PncEnrollmentController@confirm (needs permission:evaluation.manage)
- `PUT /v1/pnc-enrollments/{pncEnrollment}/schedule` → PncEnrollmentController@updateSchedule (needs permission:evaluation.manage)
- `POST /v1/pnc-enrollments/{pncEnrollment}/cancel` → PncEnrollmentController@cancel (needs permission:evaluation.manage)
- `GET /v1/pnc-enrollments/{pncEnrollment}` → PncEnrollmentController@show

### Patient evaluation service — `pnc-visit-schedules` endpoints

Area: wildcard so they aren't swallowed by it.

- `POST /v1/pnc-visit-schedules/{pncVisitSchedule}/check-in` → PncVisitScheduleController@checkIn (needs permission:evaluation.manage)

### Patient evaluation service — `prescription-frequencies` endpoints

Area: Prescription Lookup Options

- `GET /v1/prescription-frequencies` → PrescriptionFrequencyController@index
- `POST /v1/prescription-frequencies` → PrescriptionFrequencyController@store
- `GET /v1/prescription-frequencies/{id}` → PrescriptionFrequencyController@show
- `PUT/PATCH /v1/prescription-frequencies/{id}` → PrescriptionFrequencyController@update
- `DELETE /v1/prescription-frequencies/{id}` → PrescriptionFrequencyController@destroy

### Patient evaluation service — `prescription-measures` endpoints

Area: Prescription Lookup Options

- `GET /v1/prescription-measures` → PrescriptionMeasureController@index
- `POST /v1/prescription-measures` → PrescriptionMeasureController@store
- `GET /v1/prescription-measures/{id}` → PrescriptionMeasureController@show
- `PUT/PATCH /v1/prescription-measures/{id}` → PrescriptionMeasureController@update
- `DELETE /v1/prescription-measures/{id}` → PrescriptionMeasureController@destroy

### Patient evaluation service — `prescription-routes` endpoints

Area: Prescription Lookup Options

- `GET /v1/prescription-routes` → PrescriptionRouteController@index
- `POST /v1/prescription-routes` → PrescriptionRouteController@store
- `GET /v1/prescription-routes/{id}` → PrescriptionRouteController@show
- `PUT/PATCH /v1/prescription-routes/{id}` → PrescriptionRouteController@update
- `DELETE /v1/prescription-routes/{id}` → PrescriptionRouteController@destroy

### Patient evaluation service — `prescriptions` endpoints

Area: (edit_system_timestamp / adjusted_date).; Prescription Management — fixed-segment routes MUST come before apiResource; from the general update() route.; gateway, so the same permission gate as a direct dispense applies.; instead of one `visitHasPharmacyBill` call per row (N+1 across services).; sale's own drug lines paid. Service-to-service; nothing here touches stock.

- `GET /v1/prescriptions/own` → PrescriptionController@own
- `GET /v1/prescriptions/cancelled` → PrescriptionController@cancelled
- `GET /v1/prescriptions/administrations` → PrescriptionController@administrations
- `GET /v1/prescriptions/pharmacy-bill-status` → PrescriptionController@pharmacyBillStatus
- `POST /v1/prescriptions/payable-check` → PrescriptionController@payableCheck
- `POST /v1/prescriptions/fetch-return-status` → PrescriptionController@fetchReturnStatus
- `POST /v1/prescriptions/update-return-status` → PrescriptionController@updateReturnStatus (needs permission:pharmacy.dispensing.return|inventory.dispensing.return|prescriptions.dispense)
- `POST /v1/prescriptions/process-and-close` → PrescriptionController@processAndCloseMeds (needs permission:prescriptions.dispense|pharmacy.prescriptions.fill|pharmacy.dispensing.dispense|inventory.dispensing.dispense|evaluation.manage)
- `POST /v1/prescriptions/process-and-bill` → PrescriptionController@processAndBillMeds (needs permission:prescriptions.dispense|pharmacy.prescriptions.fill|pharmacy.dispensing.dispense|inventory.dispensing.dispense|evaluation.manage)
- `POST /v1/prescriptions/pos-checkout` → PrescriptionController@posCheckout (needs permission:prescriptions.dispense|pharmacy.prescriptions.fill|pharmacy.dispensing.dispense|inventory.dispensing.dispense|evaluation.manage)
- `POST /v1/prescriptions/pos-dispensed` → PrescriptionController@posDispensed (needs permission:prescriptions.dispense|pharmacy.dispensing.dispense|inventory.dispensing.dispense|evaluation.manage)
- `POST /v1/prescriptions/pos-paid` → PrescriptionController@posPaid (needs permission:payments.create|finance.manage|prescriptions.dispense|pharmacy.dispensing.dispense|inventory.dispensing.dispense|evaluation.manage)
- `GET /v1/prescriptions/{prescription}/processing-logs` → PrescriptionController@processingLogs (needs permission:prescriptions.view|prescriptions.dispense|pharmacy.prescriptions.view|pharmacy.dispensing.view|evaluation.manage)
- `POST /v1/prescriptions/{prescription}/dispense` → PrescriptionController@dispense (needs permission:prescriptions.dispense|pharmacy.dispensing.dispense|inventory.dispensing.dispense|evaluation.manage)
- `POST /v1/prescriptions/{prescription}/discontinue` → PrescriptionController@discontinue (needs permission:prescriptions.cancel|prescriptions.update|evaluation.manage)
- `POST /v1/prescriptions/{prescription}/cancel` → PrescriptionController@cancel (needs permission:prescriptions.cancel|evaluation.manage)
- `POST /v1/prescriptions/{prescription}/administer` → PrescriptionController@administer (needs permission:evaluation.manage|prescriptions.dispense)
- `PUT /v1/prescriptions/{prescription}/billing-sync` → PrescriptionController@markBillingSync (needs permission:finance.process_payments|billing.manage|prescriptions.update|evaluation.manage)
- `PATCH /v1/prescriptions/{prescription}/date` → PrescriptionController@updateDate (needs permission:prescriptions.update|evaluation.manage)
- `GET /v1/prescriptions` → PrescriptionController@index
- `POST /v1/prescriptions` → PrescriptionController@store
- `GET /v1/prescriptions/{id}` → PrescriptionController@show
- `PUT/PATCH /v1/prescriptions/{id}` → PrescriptionController@update
- `DELETE /v1/prescriptions/{id}` → PrescriptionController@destroy

### Patient evaluation service — `printing` endpoints

Area: Bulk; Clinical documents; Job status; Reports; Templates

- `GET /v1/printing/investigation/{id}` → PrintingController@printInvestigation
- `GET /v1/printing/lab-result/{id}` → PrintingController@printLabResult
- `GET /v1/printing/prescription/{id}` → PrintingController@printPrescription
- `GET /v1/printing/doctor-note/{id}` → PrintingController@printDoctorNote
- `GET /v1/printing/visit/{visitId}/summary` → PrintingController@printVisitSummary
- `GET /v1/printing/vitals/{visitId}` → PrintingController@printVitals
- `POST /v1/printing/bulk/investigations` → PrintingController@bulkInvestigations
- `POST /v1/printing/bulk/documents` → PrintingController@bulkDocuments
- `GET /v1/printing/reports/daily-investigations` → PrintingController@dailyInvestigationsReport
- `GET /v1/printing/reports/patient-history/{patientId}` → PrintingController@patientHistoryReport
- `GET /v1/printing/templates` → PrintingController@getTemplates
- `POST /v1/printing/templates` → PrintingController@createTemplate
- `PUT /v1/printing/templates/{id}` → PrintingController@updateTemplate
- `GET /v1/printing/jobs/{jobId}` → PrintingController@getJobStatus

### Patient evaluation service — `procedure-categories` endpoints

Area: Procedure Categories

- `GET /v1/procedure-categories/csv-template` → ProcedureCategoryController@csvTemplate
- `POST /v1/procedure-categories/csv-import` → ProcedureCategoryController@csvImport (needs permission:lab.setup|settings.manage)
- `GET /v1/procedure-categories` → ProcedureCategoryController@index
- `POST /v1/procedure-categories` → ProcedureCategoryController@store
- `GET /v1/procedure-categories/{id}` → ProcedureCategoryController@show
- `PUT/PATCH /v1/procedure-categories/{id}` → ProcedureCategoryController@update
- `DELETE /v1/procedure-categories/{id}` → ProcedureCategoryController@destroy

### Patient evaluation service — `procedure-prices` endpoints

Area: Procedure Prices

- `GET /v1/procedure-prices/by-procedure/{procedureId}` → ProcedureController@getPriceByProcedure
- `PUT /v1/procedure-prices/{price}` → ProcedureController@updatePrice (needs permission:billing.configure_rates|settings.manage|lab.setup)
- `DELETE /v1/procedure-prices/{price}` → ProcedureController@destroyPrice (needs permission:billing.configure_rates|settings.manage|lab.setup)

### Patient evaluation service — `procedure-tags` endpoints

Area: Procedure Tags

- `GET /v1/procedure-tags` → ProcedureTagController@index
- `GET /v1/procedure-tags/titles` → ProcedureTagController@titles
- `POST /v1/procedure-tags/titles` → ProcedureTagController@storeTitle (needs permission:settings.manage|lab.setup|evaluation.manage)
- `POST /v1/procedure-tags/by-title` → ProcedureTagController@byTitle
- `POST /v1/procedure-tags` → ProcedureTagController@store (needs permission:settings.manage|lab.setup|evaluation.manage)

### Patient evaluation service — `procedure-template-labs` endpoints

Area: Procedure Templates (lab blueprints + rich-text templates)

- `DELETE /v1/procedure-template-labs/{id}` → ProcedureTemplateController@destroyLabEntry (needs permission:settings.manage|lab.setup|evaluation.manage)

### Patient evaluation service — `procedures` endpoints

Area: (/procedures/{procedure}) would swallow these first.; LOINC for lab tests — suggested from each name, confirmed by staff.; Procedure Templates (lab blueprints + rich-text templates)

- `GET /v1/procedures/csv-template` → ProcedureController@csvTemplate
- `POST /v1/procedures/csv-import` → ProcedureController@csvImport (needs permission:lab.setup|settings.manage)
- `GET /v1/procedures/bulk-price-export` → ProcedureController@bulkPriceExport (needs permission:billing.configure_rates|settings.manage|lab.setup)
- `POST /v1/procedures/bulk-price-import` → ProcedureController@bulkPriceImport (needs permission:billing.configure_rates|settings.manage|lab.setup)
- `POST /v1/procedures/bulk-prices` → ProcedureController@bulkStorePrices (needs permission:billing.configure_rates|settings.manage|lab.setup)
- `POST /v1/procedures/upload-bulk-prices` → ProcedureController@uploadBulkPrices (needs permission:billing.configure_rates|settings.manage|lab.setup)
- `POST /v1/procedures/resolve-bulk-prices` → ProcedureController@resolveBulkPrices (needs permission:billing.configure_rates|settings.manage|lab.setup)
- `POST /v1/procedures/terminology/suggest` → ProcedureTerminologyController@suggest (needs permission:lab.setup|settings.manage)
- `POST /v1/procedures/terminology/reset` → ProcedureTerminologyController@reset (needs permission:lab.setup|settings.manage)
- `POST /v1/procedures/{procedure}/terminology` → ProcedureTerminologyController@confirm (needs permission:lab.setup|settings.manage)
- `DELETE /v1/procedures/{procedure}/terminology/suggestion` → ProcedureTerminologyController@dismiss (needs permission:lab.setup|settings.manage)
- `DELETE /v1/procedures/{procedure}/terminology` → ProcedureTerminologyController@clear (needs permission:lab.setup|settings.manage)
- `GET /v1/procedures` → ProcedureController@index
- `POST /v1/procedures` → ProcedureController@store
- `GET /v1/procedures/{id}` → ProcedureController@show
- `PUT/PATCH /v1/procedures/{id}` → ProcedureController@update
- `DELETE /v1/procedures/{id}` → ProcedureController@destroy
- `PATCH /v1/procedures/{procedure}/toggle-active` → ProcedureController@toggleActive (needs permission:lab.setup|settings.manage)
- `POST /v1/procedures/{procedure}/prices` → ProcedureController@storePrices (needs permission:billing.configure_rates|settings.manage|lab.setup)
- `GET /v1/procedures/{procedureId}/template` → ProcedureTemplateController@show
- `POST /v1/procedures/{procedureId}/template` → ProcedureTemplateController@store (needs permission:settings.manage|lab.setup|evaluation.manage)
- `GET /v1/procedures/{procedureId}/subtests` → ProcedureTemplateController@subtests

### Patient evaluation service — `queue` endpoints

Area: Note: /queue/doctor must come before /queue/{destination} to avoid wildcard matching; pharmacy roles hold none of the visit/evaluation/reception grants.

- `GET /v1/queue/doctor` → QueueController@doctorQueue
- `GET /v1/queue/{destination}` → QueueController@index
- `POST /v1/queue/{destination}/visits/{visit}/checkout` → QueueController@checkout (needs permission:visits.close|evaluation.manage|reception.manage|queue.manage|queue.remove|lab.results.record|pharmacy.dispensing.dispense|inventory.dispensing.dispense)

### Patient evaluation service — `refraction` endpoints

Area: Ophthalmology Notes Management

- `GET /v1/refraction/{patientId?}`
- `POST /refraction` → RefractionController@store (needs permission:refraction.record|evaluation.manage)

### Patient evaluation service — `refractions` endpoints

Area: Ophthalmology Notes Management

- `GET /refractions` → RefractionController@index
- `POST /refractions` → RefractionController@store
- `GET /refractions/{id}` → RefractionController@show
- `PUT/PATCH /refractions/{id}` → RefractionController@update
- `DELETE /refractions/{id}` → RefractionController@destroy

### Patient evaluation service — `reports` endpoints

Area: Lab turnaround-time report rows (consumed by reports-service)

- `GET /v1/reports/lab-tat` → LabTatReportController@index

### Patient evaluation service — `result-analytics-groups` endpoints

Area: Additional Lab Settings

- `GET /v1/result-analytics-groups` → ResultAnalyticsGroupController@index
- `POST /v1/result-analytics-groups` → ResultAnalyticsGroupController@store
- `GET /v1/result-analytics-groups/{id}` → ResultAnalyticsGroupController@show
- `PUT/PATCH /v1/result-analytics-groups/{id}` → ResultAnalyticsGroupController@update
- `DELETE /v1/result-analytics-groups/{id}` → ResultAnalyticsGroupController@destroy

### Patient evaluation service — `sample-types` endpoints

Area: Sample types & collection methods — full CRUD

- `GET /v1/sample-types` → SampleTypeController@index
- `POST /v1/sample-types` → SampleTypeController@store
- `GET /v1/sample-types/{id}` → SampleTypeController@show
- `PUT/PATCH /v1/sample-types/{id}` → SampleTypeController@update
- `DELETE /v1/sample-types/{id}` → SampleTypeController@destroy

### Patient evaluation service — `samples` endpoints

Area: Note: named sub-resource routes must come before the apiResource to avoid wildcard conflicts

- `POST /v1/samples/{sample}/add-to-sample` → SampleCollectionController@addToSample (needs permission:lab.samples.collect|lab.samples.process|lab.samples.receive|evaluation.manage)
- `POST /v1/samples/{sample}/queue` → SampleCollectionController@queue (needs permission:lab.samples.collect|lab.samples.process|lab.samples.receive|evaluation.manage)
- `POST /v1/samples/{sample}/receive` → SampleCollectionController@receive (needs permission:lab.samples.receive|lab.samples.process|evaluation.manage)
- `POST /v1/samples/{sample}/reject` → SampleCollectionController@reject (needs permission:lab.samples.reject|evaluation.manage)
- `POST /v1/samples/{sample}/recollect` → SampleCollectionController@recollect (needs permission:lab.samples.collect|lab.samples.process|evaluation.manage)
- `GET /v1/samples` → SampleCollectionController@index
- `POST /v1/samples` → SampleCollectionController@store
- `GET /v1/samples/{id}` → SampleCollectionController@show
- `PUT/PATCH /v1/samples/{id}` → SampleCollectionController@update
- `DELETE /v1/samples/{id}` → SampleCollectionController@destroy

### Patient evaluation service — `service` endpoints

Area: VerifyInternalSignedRequest's docblock.

- `POST /v1/service/organizations/{id}/seed-reference-data` → ProcedureCatalogueController@seedReferenceData

### Patient evaluation service — `service-requests` endpoints

Area: brand-new routes with no existing caller to preserve compatibility for.; §6 Phase 3.

- `GET /v2/service-requests` → ServiceRequestController@index
- `GET /v2/service-requests/{id}` → ServiceRequestController@show
- `POST /v2/service-requests` → ServiceRequestController@store

### Patient evaluation service — `settings` endpoints

Area: Settings Bulk Configs

- `GET /v1/settings/schema` → SettingsController@schema
- `POST /v1/settings/bulk-configs` → SettingsController@bulkConfigs (needs permission:settings.manage|settings.edit)

### Patient evaluation service — `sha` endpoints

Area: A minor's fingerprint consent: dispatch a match, then read it until SHA's verdict comes.; A minor's fingerprint-match verdict (biometric_match / completed).; Building it; DHA's outcome for one SHR bundle — the X-HIE-Callback named on its submission.; Declared before /visits/{visit}; Deliveries to the SHA callback endpoints, for the callbacks dashboard.

- `POST /v1/sha/callbacks/{token}/claim-status` → ShaCallbackController@claimStatus
- `POST /v1/sha/callbacks/{token}/preauth-status` → ShaCallbackController@preauthStatus
- `POST /v1/sha/callbacks/{token}/preauth-doctor-status` → ShaCallbackController@preauthStatus
- `POST /v1/sha/callbacks/{token}/authorization-status` → ShaCallbackController@authorizationStatus
- `POST /v1/sha/callbacks/{token}/biometric-match` → ShaCallbackController@biometricMatch
- `POST /v1/sha/callbacks/{token}/biometric-enrollment` → ShaCallbackController@biometricCapture
- `POST /v1/sha/callbacks/{token}/biometric-verification` → ShaCallbackController@biometricCapture
- `POST /v1/sha/callbacks/{token}/shr-bundle/{bundleId}` → ShaCallbackController@shrBundle
- `GET /v1/sha/preauths/billing-allowed` → ShaPreauthController@billingAllowed
- `GET /v1/sha/preauths/packages` → ShaPreauthController@packages
- `GET /v1/sha/preauths/options` → ShaPreauthController@options
- `POST /v1/sha/preauths` → ShaPreauthController@store
- `GET /v1/sha/preauths/elective-options` → ShaPreauthController@electiveOptions
- `GET /v1/sha/preauths/renal-options` → ShaPreauthController@renalOptions
- `POST /v1/sha/preauths/renal` → ShaPreauthController@storeRenal
- `POST /v1/sha/preauths/diagnosis-request` → ShaPreauthDiagnosisController@request
- `GET /v1/sha/preauths/diagnosis` → ShaPreauthDiagnosisController@show
- `POST /v1/sha/preauths/diagnosis` → ShaPreauthDiagnosisController@store (needs permission:clinical_history.record|evaluation.manage)
- `GET /v1/sha/preauths/renal-documents/{type}` → ShaPreauthController@previewRenalDocument
- `GET /v1/sha/preauths` → ShaPreauthController@index
- `GET /v1/sha/preauths/export` → ShaPreauthController@export (needs permission:insurance_claims.export|insurance_claims.view|finance.manage|evaluation.manage)
- `GET /v1/sha/preauths/{preauth}` → ShaPreauthController@show
- `POST /v1/sha/preauths/{preauth}/sync` → ShaPreauthController@sync
- `POST /v1/sha/biometric-matches` → ShaBiometricMatchController@store
- `GET /v1/sha/biometric-matches/{matchId}` → ShaBiometricMatchController@show
- `GET /v1/sha/biometric-enrollments` → ShaBiometricEnrollmentController@show
- `POST /v1/sha/biometric-enrollments` → ShaBiometricEnrollmentController@store
- `GET /v1/sha/biometric-enrollments/captures/{jobId}` → ShaBiometricEnrollmentController@capture
- `GET /v1/sha/shr` → ShrWorkbenchController@index
- `GET /v1/sha/shr/{visit}` → ShrWorkbenchController@show
- `POST /v1/sha/shr/{visit}/cr-id` → ShrWorkbenchController@resolveCrId
- `POST /v1/sha/shr/{visit}/open-visits` → ShrWorkbenchController@openVisits
- `POST /v1/sha/shr/{visit}/consent` → ShrWorkbenchController@requestConsent
- `POST /v1/sha/shr/{visit}/consent/verify` → ShrWorkbenchController@verifyConsent
- `POST /v1/sha/shr/{visit}/consent/resend` → ShrWorkbenchController@resendOtp
- `POST /v1/sha/shr/{visit}/consent/status` → ShrWorkbenchController@consentStatus
- `POST /v1/sha/shr/{visit}/refresh` → ShrWorkbenchController@refresh
- `GET /v1/sha/shr/{visit}/preview` → ShrWorkbenchController@preview
- `POST /v1/sha/shr/{visit}/submit` → ShrWorkbenchController@submit
- `POST /v1/sha/shr/{visit}/read-back` → ShrWorkbenchController@readBack
- `POST /v1/sha/shr/{visit}/close` → ShrWorkbenchController@close
- `POST /v1/sha/shr/{visit}/close/verify` → ShrWorkbenchController@verifyClose
- `GET /v1/sha/callback-logs` → ShaCallbackLogController@index (needs permission:insurance_claims.view|insurance_claims.submit|finance.manage|evaluation.manage)
- `GET /v1/sha/claims/tracking` → ShaClaimTrackingController@index (needs permission:insurance_claims.view|insurance_claims.submit|insurance_claims.resubmit|finance.manage|evaluation.manage)
- `GET /v1/sha/claims/tracking/export` → ShaClaimTrackingController@export (needs permission:insurance_claims.export|insurance_claims.view|finance.manage|evaluation.manage)
- `GET /v1/sha/claims/{visit}/tracking` → ShaClaimTrackingController@show (needs permission:insurance_claims.view|insurance_claims.submit|insurance_claims.resubmit|finance.manage|evaluation.manage)
- `POST /v1/sha/claims/{visit}/tracking/sync` → ShaClaimTrackingController@sync (needs permission:insurance_claims.view|insurance_claims.submit|insurance_claims.resubmit|finance.manage|evaluation.manage)
- `POST /v1/sha/claims/{visit}/tracking/payment` → ShaClaimTrackingController@recordPayment (needs permission:finance.manage|insurance_claims.submit|evaluation.manage)
- `GET /v1/sha/claims` → ShaClaimWorkbenchController@index (needs permission:insurance_claims.view|insurance_claims.submit|insurance_claims.resubmit|finance.manage|evaluation.manage)
- `GET /v1/sha/claims/{visit}` → ShaClaimWorkbenchController@show
- `GET /v1/sha/claims/{visit}/balances` → ShaClaimWorkbenchController@balances
- `GET /v1/sha/claims/{visit}/balance` → ShaClaimWorkbenchController@balance
- `GET /v1/sha/claims/{visit}/coverage` → ShaClaimWorkbenchController@coverage
- `GET /v1/sha/claims/{visit}/invoices` → ShaClaimWorkbenchController@invoices
- `POST /v1/sha/claims/{visit}/consent/otp` → ShaClaimWorkbenchController@consentOtp
- `POST /v1/sha/claims/{visit}/consent/biometric` → ShaClaimWorkbenchController@consentBiometric
- `POST /v1/sha/claims/{visit}/consent/open` → ShaClaimWorkbenchController@openClaim
- `GET /v1/sha/claims/{visit}/diagnoses` → ShaClaimWorkbenchController@claimDiagnoses
- `POST /v1/sha/claims/{visit}/diagnoses` → ShaClaimWorkbenchController@addDiagnosis
- `DELETE /v1/sha/claims/{visit}/diagnoses` → ShaClaimWorkbenchController@removeDiagnosis
- `POST /v1/sha/claims/{visit}/diagnoses/laterality` → ShaClaimWorkbenchController@setDiagnosisLaterality
- `GET /v1/sha/claims/{visit}/lines` → ShaClaimWorkbenchController@claimLines
- `POST /v1/sha/claims/{visit}/lines` → ShaClaimWorkbenchController@addLine
- `POST /v1/sha/claims/{visit}/lines/batch` → ShaClaimWorkbenchController@addLinesBatch
- `DELETE /v1/sha/claims/{visit}/lines` → ShaClaimWorkbenchController@removeLine
- `PATCH /v1/sha/claims/{visit}/lines` → ShaClaimWorkbenchController@editLine
- `GET /v1/sha/claims/{visit}/attachments` → ShaClaimWorkbenchController@claimAttachments
- `POST /v1/sha/claims/{visit}/attachments` → ShaClaimWorkbenchController@addAttachment
- `GET /v1/sha/claims/{visit}/attachments/{attachment}/file` → ShaClaimWorkbenchController@attachmentFile
- `GET /v1/sha/claims/{visit}/printouts` → ShaClaimWorkbenchController@printouts
- `GET /v1/sha/claims/{visit}/clinical-documents` → ShaClaimWorkbenchController@clinicalDocuments
- `POST /v1/sha/claims/{visit}/clinical-documents` → ShaClaimWorkbenchController@attachClinicalDocuments
- `POST /v1/sha/claims/{visit}/printouts/{type}` → ShaClaimWorkbenchController@attachPrintout
- `GET /v1/sha/claims/{visit}/prescriptions` → ShaClaimWorkbenchController@prescriptions
- `POST /v1/sha/claims/{visit}/prescriptions/send` → ShaClaimWorkbenchController@sendVisitPrescription
- `POST /v1/sha/claims/{visit}/prescriptions/{prescription}/send` → ShaClaimWorkbenchController@sendPrescription
- `POST /v1/sha/claims/{visit}/prescriptions/{prescription}/dispense` → ShaClaimWorkbenchController@retryDispense
- `GET /v1/sha/claims/{visit}/interventions/search` → ShaClaimWorkbenchController@searchInterventions
- `POST /v1/sha/claims/{visit}/interventions` → ShaClaimWorkbenchController@addIntervention
- `POST /v1/sha/claims/{visit}/interventions/retire` → ShaClaimWorkbenchController@retireIntervention
- `POST /v1/sha/claims/{visit}/interventions/restore` → ShaClaimWorkbenchController@restoreIntervention
- `POST /v1/sha/claims/{visit}/interventions/switch` → ShaClaimWorkbenchController@switchIntervention
- `POST /v1/sha/claims/{visit}/preview` → ShaClaimWorkbenchController@previewProvider
- `POST /v1/sha/claims/{visit}/discharge-consent/otp` → ShaClaimWorkbenchController@dischargeOtp
- `POST /v1/sha/claims/{visit}/discharge-consent/biometric` → ShaClaimWorkbenchController@dischargeBiometric
- `GET /v1/sha/claims/{visit}/discharge-consent/biometric` → ShaClaimWorkbenchController@dischargeBiometricStatus
- `POST /v1/sha/claims/{visit}/discharge-consent/biometric/cancel` → ShaClaimWorkbenchController@cancelDischargeBiometric
- `DELETE /v1/sha/claims/{visit}/discharge-consent` → ShaClaimWorkbenchController@forgetDischargeConsent
- `POST /v1/sha/claims/{visit}/submit` → ShaClaimWorkbenchController@submit (needs permission:insurance_claims.submit|insurance_claims.resubmit|finance.manage)
- `POST /v1/sha/claims/{visit}/resubmit` → ShaClaimWorkbenchController@resubmit (needs permission:insurance_claims.submit|insurance_claims.resubmit|finance.manage)
- `GET /v1/sha/claims/{visit}/eccif/protocols` → ShaClaimWorkbenchController@eccifProtocols
- `GET /v1/sha/claims/{visit}/eccif/contacts` → ShaClaimWorkbenchController@eccifContacts
- `POST /v1/sha/claims/{visit}/eccif/otp` → ShaClaimWorkbenchController@eccifOtp
- `POST /v1/sha/claims/{visit}/eccif/open` → ShaClaimWorkbenchController@eccifOpen
- `POST /v1/sha/claims/{visit}/eccif/protocol` → ShaClaimWorkbenchController@eccifProtocol
- `POST /v1/sha/claims/{visit}/eccif/doctor-consent/resend` → ShaClaimWorkbenchController@eccifResendDoctorConsent
- `POST /v1/sha/claims/{visit}/eccif/doctor` → ShaClaimWorkbenchController@eccifChangeDoctor
- `POST /v1/sha/claims/{visit}/eccif/identify` → ShaClaimWorkbenchController@eccifIdentify
- `POST /v1/sha/claims/{visit}/close` → ShaClaimWorkbenchController@close (needs permission:insurance_claims.submit|insurance_claims.resubmit|finance.manage)

### Patient evaluation service — `specimens` endpoints

Area: brand-new routes with no existing caller to preserve compatibility for.; §6 Phase 3.

- `GET /v2/specimens` → SpecimenController@index
- `GET /v2/specimens/{id}` → SpecimenController@show
- `POST /v2/specimens` → SpecimenController@store

### Patient evaluation service — `surveillance` endpoints

Area: §6 Phase 3.

- `GET /v2/surveillance/moh-variation-tallies` → SurveillanceTallyController@index

### Patient evaluation service — `symptoms` endpoints

Area: Symptoms Management

- `GET /v1/symptoms/search` → SymptomController@search
- `GET /v1/symptoms` → SymptomController@index
- `POST /v1/symptoms` → SymptomController@store
- `GET /v1/symptoms/{id}` → SymptomController@show
- `PUT/PATCH /v1/symptoms/{id}` → SymptomController@update
- `DELETE /v1/symptoms/{id}` → SymptomController@destroy

### Patient evaluation service — `systems-review` endpoints

Area: which reclassified every line on it — see VisitController::moveItemsToCash.

- `POST /v1/systems-review/{visit}/store` → VisitController@storeSystemsReview

### Patient evaluation service — `template-titles` endpoints

Area: Procedure Templates (lab blueprints + rich-text templates)

- `GET /v1/template-titles` → ProcedureTemplateController@titles

### Patient evaluation service — `terminology` endpoints

Area: LOINC for lab tests — suggested from each name, confirmed by staff.

- `GET /v1/terminology/{system}` → ProcedureTerminologyController@search

### Patient evaluation service — `test-remarks` endpoints

Area: Additional Lab Settings

- `GET /v1/test-remarks` → TestRemarkController@index
- `POST /v1/test-remarks` → TestRemarkController@store
- `GET /v1/test-remarks/{id}` → TestRemarkController@show
- `PUT/PATCH /v1/test-remarks/{id}` → TestRemarkController@update
- `DELETE /v1/test-remarks/{id}` → TestRemarkController@destroy

### Patient evaluation service — `test-titles` endpoints

Area: Additional Lab Settings

- `GET /v1/test-titles` → TestTitleController@index
- `POST /v1/test-titles` → TestTitleController@store
- `GET /v1/test-titles/{id}` → TestTitleController@show
- `PUT/PATCH /v1/test-titles/{id}` → TestTitleController@update
- `DELETE /v1/test-titles/{id}` → TestTitleController@destroy

### Patient evaluation service — `test-units` endpoints

Area: Lab Settings CRUD

- `GET /v1/test-units` → LabTestUnitController@index
- `POST /v1/test-units` → LabTestUnitController@store
- `GET /v1/test-units/{id}` → LabTestUnitController@show
- `PUT/PATCH /v1/test-units/{id}` → LabTestUnitController@update
- `DELETE /v1/test-units/{id}` → LabTestUnitController@destroy

### Patient evaluation service — `treatment` endpoints

Area: bill from the insurance workbench (v2's "Add extra items to visit bills").

- `POST /v1/treatment` → TreatmentController@store (needs permission:procedures.perform|procedures.request|evaluation.manage|billing.manage|finance.manage)

### Patient evaluation service — `treatment-actions` endpoints

Area: Treatment Actions (workflow routing options)

- `GET /v1/treatment-actions` → TreatmentActionController@index

### Patient evaluation service — `treatment-locations` endpoints

Area: Treatment Locations

- `GET /v1/treatment-locations/outpatient` → TreatmentLocationController@outpatient
- `GET /v1/treatment-locations` → TreatmentLocationController@index
- `POST /v1/treatment-locations` → TreatmentLocationController@store
- `GET /v1/treatment-locations/{id}` → TreatmentLocationController@show
- `PUT/PATCH /v1/treatment-locations/{id}` → TreatmentLocationController@update
- `DELETE /v1/treatment-locations/{id}` → TreatmentLocationController@destroy

### Patient evaluation service — `update-visit-destination` endpoints

Area: Visit Destinations - Simple endpoint for reception service

- `POST /v1/update-visit-destination` → VisitController@updateVisitDestination (needs permission:visits.create|visits.update|visits.close|evaluation.manage|reception.manage)

### Patient evaluation service — `vaccinations` endpoints

Area: Vaccinations

- `POST /v1/vaccinations/{vaccination}/verify` → VaccinationController@verify (needs permission:evaluation.manage)
- `POST /v1/vaccinations/save-vaccines` → VaccinationController@saveVaccines
- `GET /v1/vaccinations/saved-vaccines` → VaccinationController@savedVaccines
- `GET /v1/vaccinations` → VaccinationController@index
- `POST /v1/vaccinations` → VaccinationController@store
- `GET /v1/vaccinations/{id}` → VaccinationController@show
- `PUT/PATCH /v1/vaccinations/{id}` → VaccinationController@update
- `DELETE /v1/vaccinations/{id}` → VaccinationController@destroy

### Patient evaluation service — `visit-icd10-variations` endpoints

Area: Frontend nests the actual fields under `slit_lamp_exam`. Unwrap if present.

- `GET /visit-icd10-variations` → VisitIcd10VariationController@index
- `POST /visit-icd10-variations` → VisitIcd10VariationController@store
- `GET /visit-icd10-variations/{id}` → VisitIcd10VariationController@show
- `PUT/PATCH /visit-icd10-variations/{id}` → VisitIcd10VariationController@update
- `DELETE /visit-icd10-variations/{id}` → VisitIcd10VariationController@destroy

### Patient evaluation service — `visit-icd11-variations` endpoints

Area: Frontend nests the actual fields under `slit_lamp_exam`. Unwrap if present.

- `GET /visit-icd11-variations` → VisitIcd11CodeController@index
- `POST /visit-icd11-variations` → VisitIcd11CodeController@store
- `GET /visit-icd11-variations/{id}` → VisitIcd11CodeController@show
- `PUT/PATCH /visit-icd11-variations/{id}` → VisitIcd11CodeController@update
- `DELETE /visit-icd11-variations/{id}` → VisitIcd11CodeController@destroy

### Patient evaluation service — `visit-moh-variations` endpoints

Area: Frontend nests the actual fields under `slit_lamp_exam`. Unwrap if present.

- `GET /visit-moh-variations` → VisitMohVariationController@index
- `POST /visit-moh-variations` → VisitMohVariationController@store
- `GET /visit-moh-variations/{id}` → VisitMohVariationController@show
- `PUT/PATCH /visit-moh-variations/{id}` → VisitMohVariationController@update
- `DELETE /visit-moh-variations/{id}` → VisitMohVariationController@destroy

### Patient evaluation service — `visits` endpoints

Area: Destination-filtered endpoints; ECCIF emergency claims (scenarios 7/8); Patient Documents; The ward closes the visit when it discharges the patient (inpatient VisitCloser).; Visit Destinations - Simple endpoint for reception service; console job (no user token to send) — see VerifyInternalSignedRequest.

- `GET /v1/visits/{visitId}/last-activity` → VisitController@lastActivity
- `GET /v1/visits/{visitId}/clinical-activity` → VisitController@hasClinicalActivity
- `POST /v1/visits/{visitId}/checkout-destination` → VisitController@checkoutFromDestination (needs permission:visits.create|visits.update|visits.close|evaluation.manage|reception.manage)
- `POST /v1/visits/{visit}/start-consultation` → VisitController@startConsultation (needs permission:evaluation.visits.update|evaluation.manage)
- `POST /v1/visits/{visit}/complete-consultation` → VisitController@completeConsultation (needs permission:evaluation.visits.update|evaluation.manage)
- `GET /v1/visits/{visit}/symptoms` → VisitController@symptoms
- `GET /v1/visits/{visit}/moh-data` → VisitController@mohData
- `POST /v1/visits/{visit}/items/move-to-cash` → VisitController@moveItemsToCash
- `POST /v1/visits/{visit}/presenting-complaints` → VisitController@storePresentingComplaints
- `GET /v1/visits/{visit}/presenting-complaints` → VisitController@getPresentingComplaints
- `GET /v1/visits/{visit}/idsr-flags` → VisitController@getIdsrFlags
- `GET /v1/visits/{visit}/systems-review` → VisitController@getSystemsReview
- `POST /v1/visits/{visit}/systems-review` → VisitController@storeSystemsReview
- `GET /v1/visits/{visit}/tb-screening` → TBScreeningController@show
- `POST /v1/visits/{visit}/tb-screening` → TBScreeningController@store
- `GET /v1/visits/{visit}/ebola-screening` → EbolaScreeningController@show
- `POST /v1/visits/{visit}/ebola-screening` → EbolaScreeningController@store
- `GET /v1/visits/{visit}/examination` → VisitController@getExamination
- `POST /v1/visits/{visit}/diagnosis` → VisitController@storeDiagnosis
- `GET /v1/visits/{visit}/patient-id` → VisitController@getPatientId
- `POST /v1/visits/{visit}/checkout` → VisitController@checkout (needs permission:visits.close|evaluation.manage|reception.manage|discharge.complete|inpatient.discharge_patients)
- `GET /v1/visits/{visit}/destinations/{destination}/consumables` → VisitController@destinationConsumables
- `GET /v1/visits/{visit}/destinations/{destination}/documents` → VisitController@destinationDocuments
- `GET /v1/visits/{visit}/destinations/{destination}/investigations` → VisitController@destinationInvestigations
- `GET /v1/visits/{visit}/destinations/{destination}/prescriptions` → VisitController@destinationPrescriptions
- `GET /v1/visits/dashboard-summary` → VisitController@dashboardSummary
- `GET /v1/visits/{visit}/triage-status` → VisitController@triageStatus
- `GET /v1/visits` → VisitController@index
- `POST /v1/visits` → VisitController@store
- `GET /v1/visits/{id}` → VisitController@show
- `PUT/PATCH /v1/visits/{id}` → VisitController@update
- `DELETE /v1/visits/{id}` → VisitController@destroy
- `GET /v1/visits/{visitId}/outpatient-view-logs` → PatientHistoryController@fetchOutPatientNotesViews
- `GET /v1/visits/{visitId}/doctor-notes-logs` → PatientHistoryController@fetchDoctorNotesLog
- `GET /v1/visits/{visitId}/nurse-notes-logs` → PatientHistoryController@fetchNurseNotesLog
- `GET /v1/visits/{visitId}/prescription-view-logs` → PatientHistoryController@fetchPrescriptionViewsLogs
- `GET /v1/visits/{visitId}/investigation-result-view-logs` → PatientHistoryController@fetchInvestigationResultViewsLogs
- `GET /v1/visits/{visitId}/inpatient-history-view-logs` → PatientHistoryController@fetchInpatientHistoryViewsLogs
- `GET /v1/visits/{visit}/documents` → PatientDocumentController@index
- `POST /v1/visits/{visit}/documents` → PatientDocumentController@store (needs permission:evaluation.manage|reception.manage|patients.update)
- `GET /v1/visits/{visit}/documents/{document}` → PatientDocumentController@show
- `POST /v1/visits/{visit}/documents/{document}` → PatientDocumentController@update (needs permission:evaluation.manage|reception.manage|patients.update)
- `DELETE /v1/visits/{visit}/documents/{document}` → PatientDocumentController@destroy (needs permission:evaluation.manage|reception.manage|patients.update)

### Patient evaluation service — `vitals` endpoints

Area: Additional Vitals endpoints; Vitals Management

- `GET /v1/vitals` → VitalController@index
- `POST /v1/vitals` → VitalController@store
- `GET /v1/vitals/{id}` → VitalController@show
- `PUT/PATCH /v1/vitals/{id}` → VitalController@update
- `DELETE /v1/vitals/{id}` → VitalController@destroy
- `GET /v1/vitals/patient/{patientId}/history` → VitalController@patientHistory

## Psychiatry service API (psychiatry-service)

Covers: psychiatric episodes, forms, risk, legal status, alerts. 62 endpoints.

### Psychiatry service — `health` endpoints

- `GET /health` → HealthCheckController@check

### Psychiatry service — `psychiatry` endpoints

Area: and the second half of one custody record.; depend on any permission the refusal was about.; each with this form's entries. One call, one access-log row.; is fetched by id through an endpoint that authorises and logs.; one case file, up to three streams — see ServiceLine.; patient needs attention, and not to read the assessment.

- `GET /v1/psychiatry/patients` → PatientController@index
- `GET /v1/psychiatry/patients/{patient}/status` → PatientController@status
- `GET /v1/psychiatry/patients/{patient}/summary` → PatientController@summary
- `GET /v1/psychiatry/patients/{patient}/access-log` → PatientController@accessLog
- `GET /v1/psychiatry/patients/{patient}/history` → PatientController@history
- `GET /v1/psychiatry/patients/{patient}/episodes` → EpisodeController@index
- `GET /v1/psychiatry/patients/{patient}/care-plan` → CarePlanController@show
- `GET /v1/psychiatry/patients/{patient}/care-plan/versions` → CarePlanController@versions
- `GET /v1/psychiatry/visits/{visit}/context` → EpisodeController@visitContext
- `POST /v1/psychiatry/episodes` → EpisodeController@store
- `POST /v1/psychiatry/episodes/{episode}/close` → EpisodeController@close
- `POST /v1/psychiatry/patients/{patient}/readmit` → EpisodeController@readmit
- `GET /v1/psychiatry/episodes/{episode}/enrolments` → EpisodeController@enrolments
- `POST /v1/psychiatry/episodes/{episode}/enrolments` → EpisodeController@enrol
- `POST /v1/psychiatry/enrolments/{enrolment}/complete` → EpisodeController@completeEnrolment
- `GET /v1/psychiatry/caseload` → CaseloadController@index
- `GET /v1/psychiatry/follow-ups` → FollowUpController@index
- `POST /v1/psychiatry/follow-ups` → FollowUpController@store
- `PATCH /v1/psychiatry/follow-ups/{followUp}` → FollowUpController@update
- `POST /v1/psychiatry/follow-ups/{followUp}/complete` → FollowUpController@complete
- `POST /v1/psychiatry/follow-ups/{followUp}/cancel` → FollowUpController@cancel
- `POST /v1/psychiatry/follow-ups/{followUp}/reschedule` → FollowUpController@reschedule
- `POST /v1/psychiatry/break-glass` → BreakGlassController@store
- `GET /v1/psychiatry/break-glass` → BreakGlassController@index
- `GET /v1/psychiatry/break-glass/{grant}` → BreakGlassController@show
- `POST /v1/psychiatry/break-glass/{grant}/review` → BreakGlassController@review
- `GET /v1/psychiatry/alerts` → AlertController@index
- `GET /v1/psychiatry/alerts/count` → AlertController@count
- `POST /v1/psychiatry/alerts/{uuid}/acknowledge` → AlertController@acknowledge
- `POST /v1/psychiatry/alerts/{uuid}/escalate` → AlertController@escalate
- `POST /v1/psychiatry/alerts/{uuid}/resolve` → AlertController@resolve
- `GET /v1/psychiatry/admission-readiness` → ReadinessController@admission
- `GET /v1/psychiatry/discharge-readiness` → ReadinessController@discharge
- `POST /v1/psychiatry/admission-forms/{uuid}/sign` → AdmissionFormController@sign
- `POST /v1/psychiatry/admission-forms/{uuid}/decision` → AdmissionFormController@decide
- `POST /v1/psychiatry/admission-forms/{uuid}/notice` → AdmissionFormController@notice
- `POST /v1/psychiatry/admission-forms/{uuid}/printed` → AdmissionFormController@printed
- `POST /v1/psychiatry/surrendered-items/{uuid}/return` → SurrenderedItemsController@returnItems
- `POST /v1/psychiatry/files` → FileController@store
- `GET /v1/psychiatry/files/{subjectType}/{subjectUuid}/{documentId}` → FileController@show
- `GET /v1/psychiatry/referrals` → ReferralController@index
- `POST /v1/psychiatry/referrals/{uuid}/send` → ReferralController@send
- `POST /v1/psychiatry/referrals/{uuid}/accept` → ReferralController@accept
- `POST /v1/psychiatry/referrals/{uuid}/decline` → ReferralController@decline
- `POST /v1/psychiatry/referrals/{uuid}/complete` → ReferralController@complete
- `POST /v1/psychiatry/referrals/{uuid}/cancel` → ReferralController@cancel
- `GET /v1/psychiatry/reports/caseload` → ReportController@caseload
- `GET /v1/psychiatry/reports/admissions-by-legal-status` → ReportController@admissionsByLegalStatus
- `GET /v1/psychiatry/reports/discharges` → ReportController@discharges
- `GET /v1/psychiatry/reports/readmissions` → ReportController@readmissions
- `GET /v1/psychiatry/reports/break-glass` → ReportController@breakGlass
- `GET /v1/psychiatry/forms/{slug}` → FormController@index
- `GET /v1/psychiatry/forms/{slug}/prefill` → FormController@prefill
- `POST /v1/psychiatry/forms/{slug}` → FormController@store
- `PATCH /v1/psychiatry/forms/{slug}/{uuid}` → FormController@update
- `POST /v1/psychiatry/forms/{slug}/{uuid}/sign` → FormController@sign
- `POST /v1/psychiatry/forms/{slug}/{uuid}/signatures` → FormController@signature
- `POST /v1/psychiatry/forms/{slug}/{uuid}/revisions` → FormController@revise
- `POST /v1/psychiatry/forms/{slug}/{uuid}/void` → FormController@void
- `GET /v1/psychiatry/forms/{slug}/{uuid}/versions` → FormController@versions
- `GET /v1/psychiatry/forms/{slug}/{uuid}/print` → FormController@print

## Reception service API (reception-service)

Covers: patients, check-in, visits, queues, appointments, follow-ups, documents. 152 endpoints.

### Reception service — `appointment-categories` endpoints

Area: Appointment Management

- `GET /v1/appointment-categories` → AppointmentCategoryController@index
- `POST /v1/appointment-categories` → AppointmentCategoryController@store
- `GET /v1/appointment-categories/{id}` → AppointmentCategoryController@show
- `PUT/PATCH /v1/appointment-categories/{id}` → AppointmentCategoryController@update
- `DELETE /v1/appointment-categories/{id}` → AppointmentCategoryController@destroy

### Reception service — `appointments` endpoints

Area: Appointment Management

- `GET /v1/appointments` → AppointmentController@index
- `POST /v1/appointments` → AppointmentController@store
- `GET /v1/appointments/{id}` → AppointmentController@show
- `PUT/PATCH /v1/appointments/{id}` → AppointmentController@update
- `DELETE /v1/appointments/{id}` → AppointmentController@destroy

### Reception service — `biometrics` endpoints

Area: beneficiary has approved by OTP — see ShaController::patientLookup.

- `GET /v1/biometrics/reader-config` → ShaController@readerConfig
- `POST /v1/biometrics/enrol` → ShaController@enrolFinger
- `POST /v1/biometrics/verify` → ShaController@verifyFinger

### Reception service — `calendarEvents` endpoints

Area: Appointment Management

- `GET /v1/calendarEvents` → AppointmentController@calendarEvents

### Reception service — `customers` endpoints

Area: Customer/Business Customer Management

- `GET /v1/customers` → CustomerController@index
- `POST /v1/customers` → CustomerController@store
- `GET /v1/customers/{id}` → CustomerController@show
- `PUT/PATCH /v1/customers/{id}` → CustomerController@update
- `DELETE /v1/customers/{id}` → CustomerController@destroy

### Reception service — `documents` endpoints

Area: Document Management - Global endpoints

- `GET /v1/documents/expiring` → PatientDocumentController@expiring
- `GET /v1/documents/expired` → PatientDocumentController@expired
- `GET /v1/documents` → PatientDocumentController@byVisit

### Reception service — `encounters` endpoints

Area: already shares read_patients with real callers depending on it.; there is no human user session on these requests. See EXTERNAL_API.md.

- `GET /v2/encounters` → EncounterController@index
- `GET /v2/encounters/{id}` → EncounterController@show
- `POST /v2/encounters` → EncounterController@store

### Reception service — `followups` endpoints

Area: Patient Follow-ups

- `GET /v1/followups` → PatientFollowupController@index
- `POST /v1/followups` → PatientFollowupController@store
- `GET /v1/followups/{id}` → PatientFollowupController@show
- `PUT/PATCH /v1/followups/{id}` → PatientFollowupController@update
- `DELETE /v1/followups/{id}` → PatientFollowupController@destroy

### Reception service — `gateway` endpoints

Area: Model Gateway — controlled cross-tenant read/insert/copy (superadmin only)

- `POST /v1/gateway` → ModelGatewayController@handle

### Reception service — `health` endpoints

Area: Health check endpoint (unversioned, public)

- `GET /health` → HealthController@index

### Reception service — `internal` endpoints

Area: ReceptionServiceClient there and VerifyInternalSignedRequest here).

- `GET /v1/internal/walkin-customer/exists` → WalkinCustomerController@exists

### Reception service — `patient-check-in` endpoints

Area: Patient Check-in Flow

- `POST /v1/patient-check-in` → PatientCheckInController@store (needs permission:reception.visits.create|visits.create|reception.manage)

### Reception service — `patients` endpoints

Area: Patient Dependant Management (Nested under patients); Patient Document Management (Nested under patients); Patient Insurance Management (Nested under patients); Patient Management; Patient Next of Kin Management (Nested under patients); Who answers for a patient (a newborn's mother), and linking a child to a guardian.

- `GET /v1/patients/search` → PatientController@search
- `POST /v1/patients/{patient}/consent` → PatientController@storeConsent
- `GET /v1/patients/{patient}/consent` → PatientController@getConsent
- `GET /v1/patients/{patient}/full-name` → PatientController@fullName
- `GET /v1/patients/{patient}/guardians` → PatientGuardianController@index
- `POST /v1/patients/{patient}/dependants/link` → PatientGuardianController@link
- `POST /v1/patients/batch-names` → PatientController@batchNames
- `POST /v1/patients/find-by-barcode` → PatientController@findByBarcode
- `GET /v1/patients/{patient}/visits` → PatientController@visits
- `GET /v1/patients/{patient}/prescriptions` → PatientController@prescriptions
- `GET /v1/patients/{patient}/investigations` → PatientController@investigations
- `GET /v1/patients/{patient}/procedures` → PatientController@procedures
- `GET /v1/patients/{patient}/doctor-notes` → PatientController@doctorNotes
- `GET /v1/patients/{patient}/finance/overview` → PatientController@financeOverview
- `GET /v1/patients/{patient}/finance/deposits` → PatientController@financeDeposits
- `GET /v1/patients/{patient}/finance/withdrawals` → PatientController@financeWithdrawals
- `GET /v1/patients/{patient}/finance/cash-payments` → PatientController@financeCashPayments
- `GET /v1/patients/{patient}/finance/invoices` → PatientController@financeInvoices
- `GET /v1/patients/{patient}/finance/insurance-claims` → PatientController@financeInsuranceClaims
- `GET /v1/patients/{patient}/finance/receipts` → PatientController@financeReceipts
- `GET /v1/patients/{patient}/finance/invoice-payments` → PatientController@financeInvoicePayments
- `GET /v1/patients/{patient}/finance/bills` → PatientController@financeBills
- `GET /v1/patients/{patient}/finance/credits` → PatientController@financeCredits
- `GET /v1/patients/{patient}/finance/waivers` → PatientController@financeWaivers
- `GET /v1/patients/{patient}/vitals` → PatientController@vitals
- `GET /v1/patients/{patient}/barcode` → PatientController@barcode
- `POST /v1/patients/{patient}/regenerate-barcode` → PatientController@regenerateBarcode (needs permission:reception.patients.update|patients.update|reception.manage)
- `POST /v1/patients/{patient}/upload-image` → PatientController@uploadImage (needs permission:reception.patients.update|patients.update|reception.manage)
- `POST /v1/patients/{patient}/random-note/store` → PatientRandomNoteController@store
- `GET /v1/patients/{patient}/samples` → PatientSampleController@patientSamples
- `GET /v1/patients/{patient}/old-system-history` → OldSystemHistoryController@index
- `GET /v1/patients` → PatientController@index
- `POST /v1/patients` → PatientController@store
- `GET /v1/patients/{id}` → PatientController@show
- `PUT/PATCH /v1/patients/{id}` → PatientController@update
- `DELETE /v1/patients/{id}` → PatientController@destroy
- `GET /v1/patients/{patient}/next-of-kin` → NextOfKinController@index
- `POST /v1/patients/{patient}/next-of-kin` → NextOfKinController@store (needs permission:reception.patients.update|patients.update|reception.manage)
- `GET /v1/patients/{patient}/next-of-kin/{nextOfKin}` → NextOfKinController@show
- `PUT /v1/patients/{patient}/next-of-kin/{nextOfKin}` → NextOfKinController@update (needs permission:reception.patients.update|patients.update|reception.manage)
- `DELETE /v1/patients/{patient}/next-of-kin/{nextOfKin}` → NextOfKinController@destroy (needs permission:reception.patients.update|patients.update|reception.manage)
- `POST /v1/patients/{patient}/next-of-kin/{nextOfKin}/set-primary` → NextOfKinController@setPrimary (needs permission:reception.patients.update|patients.update|reception.manage)
- `POST /v1/patients/{patient}/next-of-kin/{nextOfKin}/set-emergency` → NextOfKinController@setEmergencyContact (needs permission:reception.patients.update|patients.update|reception.manage)
- `GET /v1/patients/{patient}/dependants` → PatientDependantController@index
- `POST /v1/patients/{patient}/dependants` → PatientDependantController@store (needs permission:reception.patients.update|patients.update|reception.manage)
- `GET /v1/patients/{patient}/dependants/{dependant}` → PatientDependantController@show
- `PUT /v1/patients/{patient}/dependants/{dependant}` → PatientDependantController@update (needs permission:reception.patients.update|patients.update|reception.manage)
- `DELETE /v1/patients/{patient}/dependants/{dependant}` → PatientDependantController@destroy (needs permission:reception.patients.update|patients.update|reception.manage)
- `POST /v1/patients/{patient}/dependants/{dependant}/activate` → PatientDependantController@activate (needs permission:reception.patients.update|patients.update|reception.manage)
- `POST /v1/patients/{patient}/dependants/{dependant}/deactivate` → PatientDependantController@deactivate (needs permission:reception.patients.update|patients.update|reception.manage)
- `PUT /v1/patients/{patient}/dependants/{dependant}/insurance` → PatientDependantController@updateInsurance (needs permission:reception.patients.update|patients.update|reception.manage)
- `GET /v1/patients/{patient}/insurance` → PatientInsuranceController@index
- `POST /v1/patients/{patient}/insurance` → PatientInsuranceController@store (needs permission:reception.patients.update|patients.update|reception.manage)
- `GET /v1/patients/{patient}/insurance/{insurance}` → PatientInsuranceController@show
- `PUT /v1/patients/{patient}/insurance/{insurance}` → PatientInsuranceController@update (needs permission:reception.patients.update|patients.update|reception.manage)
- `DELETE /v1/patients/{patient}/insurance/{insurance}` → PatientInsuranceController@destroy (needs permission:reception.patients.update|patients.update|reception.manage)
- `POST /v1/patients/{patient}/insurance/{insurance}/set-primary` → PatientInsuranceController@setPrimary (needs permission:reception.patients.update|patients.update|reception.manage)
- `POST /v1/patients/{patient}/insurance/{insurance}/activate` → PatientInsuranceController@activate (needs permission:reception.patients.update|patients.update|reception.manage)
- `POST /v1/patients/{patient}/insurance/{insurance}/deactivate` → PatientInsuranceController@deactivate (needs permission:reception.patients.update|patients.update|reception.manage)
- `POST /v1/patients/{patient}/insurance/{insurance}/verify` → PatientInsuranceController@verify (needs permission:insurance.verify|reception.manage)
- `GET /v1/patients/{patient}/documents` → PatientDocumentController@index
- `POST /v1/patients/{patient}/documents` → PatientDocumentController@store (needs permission:reception.patients.update|patients.update|reception.manage)
- `GET /v1/patients/{patient}/documents/{document}` → PatientDocumentController@show
- `POST /v1/patients/{patient}/documents/{document}` → PatientDocumentController@update (needs permission:reception.patients.update|patients.update|reception.manage)
- `DELETE /v1/patients/{patient}/documents/{document}` → PatientDocumentController@destroy (needs permission:reception.patients.update|patients.update|reception.manage)
- `GET /v1/patients/{patient}/documents/{document}/download` → PatientDocumentController@download
- `POST /v1/patients/{patient}/documents/{document}/verify` → PatientDocumentController@verify (needs permission:reception.patients.update|patients.update|reception.manage)
- `GET /v2/patients` → PatientDemographicsController@index
- `GET /v2/patients/{id}` → PatientDemographicsController@show
- `POST /v2/patients` → PatientDemographicsController@store
- `GET /v2/patients/{patientId}/consent` → ConsentController@show
- `POST /v2/patients/{patientId}/consent` → ConsentController@store

### Reception service — `queue` endpoints

Area: Queue Management

- `POST /v1/queue/{queue}/call` → QueueController@callNext (needs permission:queue.call_next|queue.manage|reception.queue.manage|reception.manage)
- `POST /v1/queue/{queue}/start` → QueueController@startService (needs permission:queue.manage|reception.queue.manage|reception.manage)
- `POST /v1/queue/{queue}/complete` → QueueController@complete (needs permission:queue.manage|reception.queue.manage|reception.manage)
- `POST /v1/queue/{queue}/skip` → QueueController@skip (needs permission:queue.manage|reception.queue.manage|reception.manage)
- `GET /v1/queue` → QueueController@index
- `POST /v1/queue` → QueueController@store
- `GET /v1/queue/{id}` → QueueController@show
- `PUT/PATCH /v1/queue/{id}` → QueueController@update
- `DELETE /v1/queue/{id}` → QueueController@destroy

### Reception service — `samples` endpoints

Area: Patient Sample Management

- `POST /v1/samples/{patientSample}/update-result` → PatientSampleController@updateResult (needs permission:lab.results.record|lab.results.update|investigations.record_results)
- `GET /v1/samples` → PatientSampleController@index
- `POST /v1/samples` → PatientSampleController@store
- `GET /v1/samples/{id}` → PatientSampleController@show
- `PUT/PATCH /v1/samples/{id}` → PatientSampleController@update
- `DELETE /v1/samples/{id}` → PatientSampleController@destroy

### Reception service — `setup` endpoints

Area: Visit Management

- `GET /v1/setup/treatment-actions/visit/{visit}` → VisitController@treatmentActionsByVisit

### Reception service — `sha` endpoints

Area: A biometric-only beneficiary whose scans failed: ask SHA to open OTP to them.; Cheap probe the check-in form fires before showing anything SHA.; beneficiaries SHA has whitelisted for it.; beneficiary has approved by OTP — see ShaController::patientLookup.; practitioner registry used to verify a clinician's registration.

- `GET /v1/sha/is-scheme` → ShaController@isScheme
- `GET /v1/sha/eligibility` → ShaController@eligibility
- `GET /v1/sha/patient-lookup` → ShaController@patientLookup
- `GET /v1/sha/intervention-packages` → ShaController@interventionPackages
- `GET /v1/sha/facility-search` → ShaController@facilitySearch
- `GET /v1/sha/practitioner-search` → ShaController@practitionerSearch
- `POST /v1/sha/send-otp` → ShaController@sendOtp
- `POST /v1/sha/otp-whitelist` → ShaController@requestOtpWhitelist
- `POST /v1/sha/authorize-biometric` → ShaController@authorizeBiometric
- `GET /v1/sha/authorization-status` → ShaController@authorizationStatus
- `POST /v1/sha/authorization-reject` → ShaController@rejectAuthorization
- `POST /v1/sha/registry-consent/otp` → ShaController@registryConsentOtp
- `POST /v1/sha/registry-consent/verify` → ShaController@registryConsentVerify

### Reception service — `visits` endpoints

Area: Visit Management

- `GET /v1/visits/precharges` → VisitController@batchPrecharges
- `GET /v1/visits/{visit}/receipt` → VisitController@receipt
- `DELETE /v1/visits/{visit}/cancel` → VisitController@cancel (needs permission:visits.close|visits.delete|reception.manage)
- `POST /v1/visits/{visit}/check-in` → VisitController@checkIn (needs permission:reception.visits.create|visits.create|reception.manage)
- `POST /v1/visits/{visit}/check-out` → VisitController@checkOut (needs permission:visits.close|reception.manage)
- `POST /v1/visits/{visit}/change-destination` → VisitController@changeDestination (needs permission:visits.refer|queue.transfer|reception.manage)
- `GET /v1/visits/{visit}/check-capitation` → VisitController@checkCapitation
- `GET /v1/visits/{patient}/last-visit-meta` → VisitController@lastVisitMeta
- `GET /v1/visits/{visit}/destinations` → VisitController@getDestinations
- `POST /v1/visits/{visit}/destinations` → VisitController@addDestinations (needs permission:visits.refer|queue.transfer|reception.manage)
- `POST /v1/visits/{visit}/destinations/{destination}/checkout` → VisitController@checkoutDestination (needs permission:visits.refer|queue.transfer|reception.manage)
- `GET /v1/visits` → VisitController@index
- `POST /v1/visits` → VisitController@store
- `GET /v1/visits/{id}` → VisitController@show
- `PUT/PATCH /v1/visits/{id}` → VisitController@update
- `DELETE /v1/visits/{id}` → VisitController@destroy

### Reception service — `walkin-customer` endpoints

Area: (see WalkinCustomerController's own docblock).

- `POST /v1/walkin-customer` → WalkinCustomerController@store (needs permission:settings.manage|pos.setup|reception.manage)

## Reports service API (reports-service)

Covers: report definitions and report data. 5 endpoints.

### Reports service — `catalog` endpoints

Area: global prepend (bootstrap/app.php), so it no longer needs listing here.

- `GET /catalog` → ReportsController@catalog

### Reports service — `health` endpoints

- `GET /health`
- `GET /health`

### Reports service — `measure-reports` endpoints

Area: user session, only a registered signed app.

- `GET /measure-reports/{codePrefix}` → MeasureReportController@show

### Reports service — `module` endpoints

Area: global prepend (bootstrap/app.php), so it no longer needs listing here.

- `GET /module/{module}/{key}/metadata` → ReportsController@metadata

## Theatre service API (theatre-service)

Covers: theatre bookings, schedules, ward checks, operations. 94 endpoints.

### Theatre service — `gateway` endpoints

Area: Model Gateway

- `POST /v1/gateway` → ModelGatewayController@handle

### Theatre service — `health` endpoints

Area: Health check endpoint (public, unversioned)

- `GET /health` → HealthCheckController@check

### Theatre service — `procedures` endpoints

Area: Legacy compatibility aliases (procedures -> theatre-operations)

- `GET /v1/procedures` → TheatreOperationController@index
- `POST /v1/procedures` → TheatreOperationController@store
- `GET /v1/procedures/{id}` → TheatreOperationController@show
- `PUT/PATCH /v1/procedures/{id}` → TheatreOperationController@update
- `DELETE /v1/procedures/{id}` → TheatreOperationController@destroy
- `POST /v1/procedures/{id}/start` → TheatreOperationController@start (needs permission:theatre.operations.start)
- `POST /v1/procedures/{id}/complete` → TheatreOperationController@complete (needs permission:theatre.operations.complete)
- `POST /v1/procedures/{id}/vitals` → TheatreOperationController@addVitals (needs permission:theatre.operation.record-vitals)
- `POST /v1/procedures/{id}/fluids` → TheatreOperationController@addFluid (needs permission:theatre.operation.record-vitals)
- `POST /v1/procedures/{id}/investigations` → TheatreOperationController@addInvestigation (needs permission:theatre.operation.record-vitals)

### Theatre service — `theatre-bookings` endpoints

Area: Theatre Bookings Management; anaesthetist.; the broader `view` permission so a nurse can still see the other checks.; the surgeon or the anaesthetist.

- `GET /v1/theatre-bookings` → TheatreBookingController@index
- `POST /v1/theatre-bookings` → TheatreBookingController@store
- `GET /v1/theatre-bookings/{id}` → TheatreBookingController@show
- `PUT/PATCH /v1/theatre-bookings/{id}` → TheatreBookingController@update
- `DELETE /v1/theatre-bookings/{id}` → TheatreBookingController@destroy
- `POST /v1/theatre-bookings/{id}/approve` → TheatreBookingController@approve (needs permission:theatre.bookings.approve)
- `POST /v1/theatre-bookings/{id}/reject` → TheatreBookingController@reject (needs permission:theatre.bookings.reject|theatre.bookings.approve)
- `POST /v1/theatre-bookings/{id}/cancel` → TheatreBookingController@cancel (needs permission:theatre.bookings.cancel|theatre.bookings.approve)
- `POST /v1/theatre-bookings/{id}/authorize` → TheatreBookingController@authorizePayment (needs permission:theatre.bookings.authorize-procedure|theatre.bookings.approve)
- `POST /v1/theatre-bookings/{bookingId}/pre-op-checks/anaesthetist` → PreOperativeCheckController@storeAnaesthetistCheck
- `POST /v1/theatre-bookings/{bookingId}/pre-op-checks/doctor` → PreOperativeCheckController@storeDoctorCheck
- `POST /v1/theatre-bookings/{bookingId}/pre-op-checks/nurse` → PreOperativeCheckController@storeNurseCheck
- `GET /v1/theatre-bookings/{bookingId}/pre-op-checks` → PreOperativeCheckController@getChecks
- `GET /v1/theatre-bookings/{bookingId}/consent` → TheatreConsentController@show
- `POST /v1/theatre-bookings/{bookingId}/consent` → TheatreConsentController@store
- `POST /v1/theatre-bookings/{bookingId}/consent/sign/patient` → TheatreConsentController@signPatient
- `POST /v1/theatre-bookings/{bookingId}/consent/sign/witness` → TheatreConsentController@signWitness
- `POST /v1/theatre-bookings/{bookingId}/consent/sign/surgeon` → TheatreConsentController@signSurgeon
- `POST /v1/theatre-bookings/{bookingId}/consent/sign/anaesthetist` → TheatreConsentController@signAnaesthetist
- `POST /v1/theatre-bookings/{bookingId}/consent/withdraw` → TheatreConsentController@withdraw

### Theatre service — `theatre-consumable-requirements` endpoints

Area: evaluation/pharmacy/dialysis pages by roles with no theatre.* grants.

- `GET /v1/theatre-consumable-requirements` → TheatreConsumableRequirementController@index
- `POST /v1/theatre-consumable-requirements` → TheatreConsumableRequirementController@store
- `GET /v1/theatre-consumable-requirements/{id}` → TheatreConsumableRequirementController@show
- `PUT/PATCH /v1/theatre-consumable-requirements/{id}` → TheatreConsumableRequirementController@update
- `DELETE /v1/theatre-consumable-requirements/{id}` → TheatreConsumableRequirementController@destroy

### Theatre service — `theatre-deposits` endpoints

Area: Theatre Deposits Management

- `GET /v1/theatre-deposits` → TheatreDepositController@index
- `POST /v1/theatre-deposits` → TheatreDepositController@store
- `GET /v1/theatre-deposits/{id}` → TheatreDepositController@show
- `PUT/PATCH /v1/theatre-deposits/{id}` → TheatreDepositController@update
- `DELETE /v1/theatre-deposits/{id}` → TheatreDepositController@destroy

### Theatre service — `theatre-finance` endpoints

Area: which calls these over the intermodular gateway.

- `GET /v1/theatre-finance/billable` → TheatreFinanceController@billable

### Theatre service — `theatre-operation-medics` endpoints

Area: Theatre Operation Team/Medics Management

- `GET /v1/theatre-operation-medics` → TheatreOperationMedicController@index
- `POST /v1/theatre-operation-medics` → TheatreOperationMedicController@store
- `GET /v1/theatre-operation-medics/{id}` → TheatreOperationMedicController@show
- `PUT/PATCH /v1/theatre-operation-medics/{id}` → TheatreOperationMedicController@update
- `DELETE /v1/theatre-operation-medics/{id}` → TheatreOperationMedicController@destroy

### Theatre service — `theatre-operations` endpoints

Area: Theatre Operations Management; WHO Surgical Safety Checklists; sheet above.; the controller.; which calls these over the intermodular gateway.

- `GET /v1/theatre-operations` → TheatreOperationController@index
- `POST /v1/theatre-operations` → TheatreOperationController@store
- `GET /v1/theatre-operations/{id}` → TheatreOperationController@show
- `PUT/PATCH /v1/theatre-operations/{id}` → TheatreOperationController@update
- `DELETE /v1/theatre-operations/{id}` → TheatreOperationController@destroy
- `POST /v1/theatre-operations/{id}/start` → TheatreOperationController@start (needs permission:theatre.operations.start)
- `POST /v1/theatre-operations/{id}/complete` → TheatreOperationController@complete (needs permission:theatre.operations.complete)
- `POST /v1/theatre-operations/{id}/send-to-morgue` → TheatreOperationController@sendToMorgue (needs permission:morgue.intake.request|theatre.operations.complete)
- `POST /v1/theatre-operations/{id}/vitals` → TheatreOperationController@addVitals (needs permission:theatre.operation.record-vitals)
- `POST /v1/theatre-operations/{id}/fluids` → TheatreOperationController@addFluid (needs permission:theatre.operation.record-vitals)
- `POST /v1/theatre-operations/{id}/investigations` → TheatreOperationController@addInvestigation (needs permission:theatre.operation.record-vitals)
- `GET /v1/theatre-operations/{operation}/billing-data` → TheatreFinanceController@billingData
- `POST /v1/theatre-operations/{operation}/mark-billed` → TheatreFinanceController@markBilled
- `GET /v1/theatre-operations/{operation}/daily-charges` → TheatreDailyChargeController@index
- `POST /v1/theatre-operations/{operation}/daily-charges` → TheatreDailyChargeController@store
- `PUT /v1/theatre-operations/{operation}/daily-charges/{charge}` → TheatreDailyChargeController@update
- `POST /v1/theatre-operations/{operation}/daily-charges/{charge}/approve` → TheatreDailyChargeController@approve
- `DELETE /v1/theatre-operations/{operation}/daily-charges/{charge}` → TheatreDailyChargeController@destroy
- `GET /v1/theatre-operations/{operation}/daily-charges/verifications` → TheatreDailyChargeController@verifications
- `POST /v1/theatre-operations/{operation}/daily-charges/verify-day` → TheatreDailyChargeController@verifyDayBills
- `POST /v1/theatre-operations/{operationId}/who-checklist/before-anaesthesia` → WHOChecklistController@storeBeforeAnaesthesia (needs permission:theatre.who-checklist.before-anaesthesia)
- `POST /v1/theatre-operations/{operationId}/who-checklist/before-skin-incision` → WHOChecklistController@storeBeforeSkinIncision (needs permission:theatre.who-checklist.before-incision)
- `POST /v1/theatre-operations/{operationId}/who-checklist/before-leaving-theatre` → WHOChecklistController@storeBeforeLeavingTheatre (needs permission:theatre.who-checklist.before-leaving)
- `GET /v1/theatre-operations/{operationId}/who-checklist` → WHOChecklistController@getChecklists

### Theatre service — `theatre-payments` endpoints

Area: Theatre Payments Management

- `GET /v1/theatre-payments` → TheatrePaymentController@index
- `POST /v1/theatre-payments` → TheatrePaymentController@store
- `GET /v1/theatre-payments/{id}` → TheatrePaymentController@show
- `PUT/PATCH /v1/theatre-payments/{id}` → TheatrePaymentController@update
- `DELETE /v1/theatre-payments/{id}` → TheatrePaymentController@destroy

### Theatre service — `theatre-schedules` endpoints

Area: Theatre Schedules Management

- `GET /v1/theatre-schedules` → TheatreScheduleController@index
- `POST /v1/theatre-schedules` → TheatreScheduleController@store
- `GET /v1/theatre-schedules/{id}` → TheatreScheduleController@show
- `PUT/PATCH /v1/theatre-schedules/{id}` → TheatreScheduleController@update
- `DELETE /v1/theatre-schedules/{id}` → TheatreScheduleController@destroy
- `POST /v1/theatre-schedules/conflicts` → TheatreScheduleController@conflicts

### Theatre service — `theatre-types` endpoints

Area: Theatre Types Management

- `GET /v1/theatre-types` → TheatreTypeController@index
- `POST /v1/theatre-types` → TheatreTypeController@store
- `GET /v1/theatre-types/{id}` → TheatreTypeController@show
- `PUT/PATCH /v1/theatre-types/{id}` → TheatreTypeController@update
- `DELETE /v1/theatre-types/{id}` → TheatreTypeController@destroy

### Theatre service — `theatres` endpoints

Area: Theatre Rooms Management

- `GET /v1/theatres` → TheatreController@index
- `POST /v1/theatres` → TheatreController@store
- `GET /v1/theatres/{id}` → TheatreController@show
- `PUT/PATCH /v1/theatres/{id}` → TheatreController@update
- `DELETE /v1/theatres/{id}` → TheatreController@destroy
- `GET /v1/theatres/{id}/availability` → TheatreController@availability
