# Afya Analytics HMIS — Setup and User Knowledge Base

**Product:** Afya Analytics HMIS / ERP, version 3.0 (the successor to CollabMed 2.0)
**Audience:** implementation and support staff who set the system up for a client, the client's own administrators, and the hospital staff who use it every day
**Purpose:** source document for the support chatbot. Every section is written to stand on its own, so it can be retrieved and read without the sections around it.

## How to read this document

- **Menu** lines give the path in the left sidebar, for example `Setup > HIS Client Settings > Facilities`.
- **Page address** lines give the part of the web address after the client's site name, for example `/core/setup/facilities`.
- **Permission** lines give the permission code a user needs to see that screen. If a user says "I cannot see this menu", the cause is almost always a missing permission, a missing facility assignment, or a module that is switched off (see sections 3.4, 3.8 and 21).
- Words in **bold** are the exact labels on buttons, tabs and fields. A field marked with * on screen is required.
- Currency is shown in Kenya Shillings (Ksh / KES) unless the client's currency setting says otherwise.
- "Setting key" means the internal name of a configuration switch, for example `inpatient.auto_generate_inpatient_no`. Section 20 lists them.

### Where to find things in this document

| You want to know about | Go to section |
|---|---|
| What the system is and how it is organised | 1 |
| Signing in, passwords, two-step verification | 2 |
| Setting up a new client, screen by screen | 3 |
| Daily use of each module | 4 to 17 |
| Laboratory configuration | 18 |
| Reports | 19 |
| Every configuration setting | 20 |
| Every permission code and suggested roles | 21 |
| Installing and deploying the application | 22 |
| Setup checklists by type of client | 23 |
| Troubleshooting questions and answers | 24 |
| Glossary | 25 |

---

## 1. System overview

### 1.1 What the system is

Afya Analytics HMIS is a web-based hospital management system. Staff open it in a browser, sign in, and work in the modules their role allows. It covers the whole patient journey: registration, queues, triage, consultation, laboratory, pharmacy, billing, admission, theatre and discharge, plus stock, reports and specialised clinics.

Nothing is installed on staff computers for normal use. The exceptions are optional devices: a fingerprint reader and the SHA HealthID application at reception (section 3.22), and receipt or label printers.

### 1.2 Modules

| Module | Sidebar section | What it is for |
|---|---|---|
| Dashboard | Dashboard | Live summary of visits, queues, admissions, stock and revenue |
| Reception | Reception & Registration | Patient registration, check-in, appointments, follow-ups, patient documents |
| Outpatient | OutPatient | Clinical queues: triage, consultation, laboratory, pharmacy, radiology and any other destination the facility has set up |
| Inpatient | In-Patient | Admission requests, wards and beds, inpatient charting, discharge |
| MCH | MCH | Antenatal (ANC), postnatal (PNC), child welfare (CWC) and family planning clinics |
| Theatre | Theatre | Surgery bookings, scheduling, pre-operative checks, operation records |
| Oncology | Oncology | Cancer screening, oncology clinic, chemotherapy day unit, palliative care |
| Morgue | Morgue | Receiving, storing, releasing and billing for bodies |
| Renal/Dialysis | Renal/Dialysis | Dialysis enrollment, schedules and treatment sessions |
| Inventory | Inventory & Procurement | Products, stores, suppliers, requisitions, purchase orders, goods received |
| Point of Sale | Point of Sale | Over-the-counter sales with shifts and receipts |
| Psychiatry | Psychiatry & Rehab | Mental health, rehabilitation and therapeutic services with restricted records |
| Finance | Finance | Cash Workbench, Insurance Workbench, receipts, invoices, credit notes, waivers, patient accounts |
| SHA | SHA | Social Health Authority claims, preauthorisations and the Shared Health Record |
| SMS | SMS | Text messaging (shown as "Coming soon") |
| Users | Users & Staff | User accounts, roles, permissions, staff categories and specialities |
| Setup | Setup | All configuration: facilities, departments, destinations, insurance, procedures, lab, wards and more |
| Reports | Reports | Reception, outpatient, inpatient, finance, inventory, MOH, dialysis, SHA and user reports |

### 1.3 What decides whether a user sees a module

Three things must all be true for a sidebar section to appear:

1. **The section is switched on for the organization.** Each sidebar section has an on/off switch in `Setup > General Settings > Core`, under "Side menu" (section 3.8). All sections are on by default except **Morgue**, which is off by default. Dashboard, Users & Staff and Setup cannot be switched off.
2. **The user has the permission.** Each section and each screen inside it needs a permission (section 21).
3. **For some sections, a feature switch is on.** The SHA section and the SHA report group appear only when SHA integration is switched on (`sha_integration`).

When a section is switched off, its configuration screens under Setup are shown greyed out rather than hidden, so an administrator can still see that they exist.

### 1.4 Organizations, regions, facilities and departments

- An **organization** (also called a tenant) is the client: a hospital or hospital group. Each organization's data is separate from every other organization's.
- A **region** groups facilities, for example by county or area.
- A **facility** is one branch or site. A facility is either a **Main** facility or a **Branch**.
- A **department** is a unit inside the organization, such as Outpatient, Laboratory or Accounts.
- Every user is assigned to one or more facilities. Lists are filtered to the user's organization and assigned facilities automatically.
- The facility a user is currently working in is shown as a badge in the top bar. New records default to that facility.
- A person registered under more than one organization chooses the organization when signing in.

### 1.5 The screen layout

- **Left sidebar:** the modules and screens the user has permission to use.
- **Top bar:** the current facility badge, a light/dark theme switch, a **Quick Actions** search, **Notifications** (with **Mark all read**), and the profile menu.
- **Profile menu:** **My Profile**, **Change Password**, **Sign Out**.
- Most list screens have a search box, date filters, a **Filter** button, a **Rows per page** control and an **Actions** menu on each row.
- Setup screens usually show an entry form at the top or in a dialog, and the existing records in a table below, with **Edit** and **Delete** (or **Deactivate** / **Activate**) on each row.

### 1.6 Technical overview (for implementers)

- The application is a single-page web application. It talks to a set of separate backend services, one per business area: core, reception, patient evaluation, inventory, inpatient, theatre, dialysis, finance, reports, oncology, morgue, integration and psychiatry.
- Every request carries the signed-in user's token and the organization and facility it applies to.
- If one backend service is unavailable, only that module's screens fail; the rest keep working.
- Section 22 covers installation, environment configuration and deployment.

---

## 2. Signing in and passwords

### 2.1 How to sign in

1. Open the client's HMIS web address. The sign-in page is at `/auth/login`.
2. Enter the **Username** (or email) and **Password**, then click **Sign In**.
3. If the account belongs to more than one organization, pick the organization.
4. If the user is assigned to more than one facility, pick the facility they are working in today.
5. If two-step verification is switched on for the organization, enter the 6-digit code that was emailed (section 2.6).
6. The **Dashboard** opens.

The sign-in page shows the organization's own logo and colours once they have been set up (section 3.11).

### 2.2 First sign-in for a new user

There are two ways a new user gets their first password, depending on how the account was created.

**Created by an administrator on the Create User screen.** When the account is saved, a **User Created** box shows the username and a password with the note "Share these login details with the new user — the password won't be shown again." The administrator copies them and gives them to the user. If they were not copied, the administrator resets the password (section 2.5).

**Invited by email** (for example through bulk onboarding, section 3.25). The user receives a link that opens **Complete Your Setup**. They enter a **New Password** of at least 8 characters, confirm it, and click **Activate Account**. The link expires after 8 hours; if it shows "Setup link expired", click **Request new link**.

### 2.3 Forgotten password

1. On the sign-in page click **Forgot password?**
2. Enter the **Email or Username**.
3. Click **Send Reset Link** (a link is emailed) or **Send OTP** (a 6-digit code is sent; enter it and click **Verify Code**; use **Resend OTP** if it does not arrive). Which of the two is offered depends on the deployment.
4. If more than one account matches, the page asks you to **Select your organization** first.
5. On **Set a new password**, enter a password of at least 8 characters and confirm it.

If the page says "Invalid or missing reset link", click **Request a new one**.

### 2.4 Changing your own password

Open the profile menu in the top bar and choose **Change Password**. Enter the **Current Password**, then the **New Password** twice.

### 2.5 An administrator resetting someone's password

Menu: `Users & Staff > View Existing`, open the user, then the **Credentials** tab.

- **Force Reset Password** / **Hard Reset Password** resets the password; confirm with **Confirm Reset**. You can also set a specific new password for the user.
- The same tab changes the **Username** (tick **Confirm change of username**) and the **Email** (tick **Confirm change of email**).

### 2.6 Two-step verification

When two-step verification is on, signing in has an extra step. After the username, password and facility are accepted, a 6-digit code is emailed to the user. The user types it and clicks **Verify My Account**. **Didn't get the code?** sends it again. The user must have a working email address on their account for this to work.

### 2.7 Messages on the sign-in page and access pages

| Message | Meaning | What to do |
|---|---|---|
| "We couldn't complete this right now. Please try again in a moment, or contact support if this keeps happening." | The request did not reach the server, or the server could not answer. It is not a wrong-password message. | Check the internet connection and retry. If everyone at the facility is affected, the service may be down: contact support. For a new installation, check that the core service address is configured and reachable (section 22). |
| "401 — Unauthorized. Your session has expired or you don't have permission. Please log in again." | The session ended. | Click **Sign In Again**. |
| "403 — Forbidden. You don't have permission to access this resource." | The user opened a page their role does not allow. | An administrator grants the permission (sections 3.4 and 21). |
| "No dashboard widgets are available for your role." | The user can sign in but has no dashboard permissions. | Grant `dashboard.view` and the module permissions. |

### 2.8 Scheduled maintenance

Before a planned maintenance window, a notice dialog appears when the page loads and a chip in the top bar shows the window. During the window the whole application is replaced by a maintenance page and open tabs lock by themselves. When the window ends the lock lifts automatically and a "System upgrade complete" notice is shown for about 30 minutes. Printing is not affected by the notice. Section 22.6 explains how support staff schedule a window.

---

## 3. Setting up a new client

This section is the complete configuration guide. Work through it in order for a new client. Later steps depend on earlier ones: queues need destinations, billing needs procedures and prices, pharmacy needs stores and products, and users need roles and facilities.

The **Setup** landing page (`/core/setup`) lists every setup screen the signed-in user has permission for, and has a search box. Try "facilities", "insurance", "dialysis" or "POS". Menu permission for the Setup section: `settings.manage`.

### 3.1 Recommended setup order

| Step | What to set up | Section |
|---|---|---|
| 1 | Organization details, logo, letterhead, colours | 3.2, 3.11 |
| 2 | Regions, then facilities (with SHA FR code if the client uses SHA) | 3.2 |
| 3 | Departments | 3.2 |
| 4 | Sidebar sections to switch on or off; general settings | 3.8 |
| 5 | Patient registration form fields | 3.8 |
| 6 | Procedure categories, then procedures and services with prices | 3.5 |
| 7 | Treatment actions and destinations (queues) | 3.3 |
| 8 | Insurance companies, schemes, exclusions and scheme prices | 3.6 |
| 9 | Payment modes, banks, service charges, rebates | 3.7 |
| 10 | Inventory: tax categories, units, categories, brands, products, stores, suppliers, approval levels, opening stock | 3.12 |
| 11 | Points of sale and POS station authorisation | 3.13 |
| 12 | Laboratory settings | 18 |
| 13 | Wards, beds, admission types and inpatient charges | 3.14 |
| 14 | Theatre types, theatres and deposits | 3.15 |
| 15 | Dialysis, morgue, psychiatry, oncology and MCH, if used | 3.16 to 3.20 |
| 16 | SHA integration, if used | 3.21, 3.22 |
| 17 | Staff categories, specialities, roles, then users | 3.4 |
| 18 | Appointment categories, partner institutions | 3.9 |
| 19 | Test a full patient visit, then go live | 3.26 |

### 3.2 Organization, regions, facilities and departments

Menu: `Setup > HIS Client Settings`. Permission: `settings.manage_locations`, plus the view permission for each item.

#### Organization

Page address: `/core/setup/organization`. Permission: `organizations.view`.

Fields: **Organization Name** *, Code, Type, Registration No., **KRA PIN / Tax Number**, Status; contact details (Telephone, Mobile, Email, Website, Fax); address (Physical Address, Postal Address, City / Town, Country, Postal Code, Street, Building, Office Number); Notes; **Logo**; **Letterhead**. Click **Save Organization**.

- **Logo:** a square image is recommended, around 300 × 300 pixels. The logo is also what prints on 80mm thermal receipts, so keep it compact rather than a wide banner.
- **Letterhead:** a wide banner image for A4 and A5 printouts, for example about 1600 × 400 pixels. Thermal receipts never use the letterhead. PDF letterheads are not supported; upload an image.
- **Use logo as letterhead instead:** tick this if the client has no separate letterhead banner.
- Large images are compressed before saving. If you see "Selected image is too large", use a smaller image.

The organization details appear on receipts, invoices, discharge summaries and other printouts, so complete them before go-live.

#### Regions

Page address: `/core/setup/regions`. Permission: `regions.view`.

Fields: **Region Name** *, Code (optional short code), Country, Active/Inactive, Description. A region must exist before a facility can be created in it. For a single-site client, create one region.

Counties and sub-counties are managed at `/core/setup/counties`. They feed the "County of residence" and "Sub county of residence" lists on the patient form.

#### Facilities

Page address: `/core/setup/facilities`. Permission: `facilities.view`.

Click **Add Facility**. Fields:

| Field | Notes |
|---|---|
| **Name** * | The facility name shown in the top bar and on printouts |
| Code | A short code for the facility |
| **SHA FR Code** | For example `FID-12191-1`. Sent to SHA/DHA as this facility on claims, eligibility checks and the Shared Health Record. Required for SHA to work at this facility |
| Type | **Main** or **Branch** |
| **Region** * | The region the facility belongs to |
| City, Telephone, Address | Contact details |
| KEPH Level, Master Facility Code | The facility's Ministry of Health level and code |
| Bed Capacity | Number of beds |
| Description | Used on printouts |
| Logo | A facility-specific logo, with **Use logo as letterhead instead** |
| Status | Active, Inactive or Under Construction |

Rules:
- Every organization needs at least one facility.
- Mark the head-office site as **Main** and the others as **Branch**.
- Facilities are switched off with **Deactivate** and back on with **Activate**. They are not deleted.
- After adding a facility, assign users to it (section 3.4), or nobody will be able to work there.
- Destinations, stores, wards, theatres, points of sale and procedure prices are all set per facility, so a new branch needs each of these set up.

#### Departments

Page address: `/core/setup/departments`. Permission: `departments.view`.

Fields: **Name**, Code, **Type** * (Clinical, Administrative, Support, Diagnostic or Therapeutic), Status, Description.

Departments are used on staff records, on stores and in reports. Always choose a Type; it is required.

#### Hierarchy Tree

Page address: `/core/setup/hierarchy`. Permission: `hierarchy.view`. A read-only tree showing Organization » Region » Facility. Use it to check the structure after setup.

### 3.3 Destinations (queues) and treatment actions

A **destination** is a place a patient is sent during a visit: Triage, Consultation, Laboratory, Pharmacy, Radiology, a dental room and so on. Each destination is a queue. Getting destinations right is the most important part of outpatient setup.

#### Create treatment actions first

Menu: `Setup > HIS Client Settings`. Page address: `/core/setup/treatment-actions`.

A **treatment action** is a tab on the clinical screen, such as Vitals, Notes, Prescribe, Investigations or Dispense. Click **Add Action** to add one. The list shows the action's **Name** and whether it **Has State**. Standard actions are normally loaded for a new organization; add to them only when a clinic needs a specialised form.

The tabs available include, among others: Vitals, Presenting Complaints, History, Systems Review, Notes, ICD-11 diagnosis, Prescribe, Ordered Drugs, Dispensed, Order New Investigation, Ordered / Approved / Published Investigations, Machine Results, Consumables, Nursing Procedures, Treatment, Documents, Claims and Insurance, and specialised forms for eye care (Preliminary Exam, Refraction, Ophthalmologist Notes, Optical Dispensing), dental (Odontogram, Periodontal Chart, Clearance), ENT, orthopaedics, maternity (ANC Profile, Obstetric History, Maternal Examination, Postnatal Assessment), MOH registers, psychiatry and rehabilitation.

#### Create the destinations

Menu: `Setup > Facility Destinations`. Page address: `/settings/destinations/create`. Permission: `settings.manage` or `settings.view`.

Click **Add Destination** and complete:

| Field | What it does |
|---|---|
| **Select a clinic** * | The facility this destination belongs to |
| **Destination type** * | Chosen from a fixed list of types that is the same for every client. The type decides where the destination appears (see below) |
| **Destination name** * | The name staff see, for example "Triage", "Doctor Consultation", "Laboratory" |
| Icon | An icon for the sidebar entry, for example `fa fa-stethoscope` |
| **Hidden from normal queues** | The destination exists but is not shown as a queue in the sidebar |
| **Auto checkin** | The destination is pre-selected whenever a patient is checked in. Use it for Triage so every patient goes there first |
| **Is a lab queue** | Marks the laboratory. Lets users with `lab.tests.view` see it without the general clinical permission |
| **Is a pharmacy queue** | Marks a pharmacy. Lets users with `pharmacy.dispensing.view` see it without the general clinical permission |
| **Is a radiology queue** | Marks the radiology or imaging unit |
| **Connected inventory store** | Links the destination to the store it dispenses from |
| **Associated Procedure Categories** | The categories of service offered at this destination, for example Laboratory tests at the lab |
| **Treatment Actions / Evaluation Actions** | The tabs staff see when they open a patient in this queue |

Rules and behaviour:
- **The OutPatient sidebar section is built from this list.** Only destinations whose type is a room or a department are listed there. Facility units such as Inpatient, Theatre, Dialysis, MCH and the cancer centre have their own sidebar sections and are not repeated under OutPatient.
- **Theatre and Dialysis cannot be configured as destinations.** They are separate modules. The screen says so if you try.
- **Queue order in the sidebar** follows the patient's path: Triage first, then Doctor / Consultation, then OPD, then Laboratory, Pharmacy and Radiology. The system recognises these by the words in the destination's name, so name them plainly (a destination called "Consultation Room 1" is treated as a consultation queue). Other destinations follow in their sort order.
- A destination that should receive lab orders must be marked **Is a lab queue**; otherwise `Setup > Lab Settings > Laboratory Location` reports "No destination is configured as the Laboratory for this facility yet."
- Use **Deactivate** / **Activate** on a row to switch a destination off and on. Deleting is available but deactivating is safer once the destination has been used.
- Psychiatry, Rehabilitation and Therapeutic queues are shown only to users with the matching psychiatry permission (section 3.18).

#### A typical minimum set for an outpatient clinic

| Destination | Flags | Typical treatment actions |
|---|---|---|
| Triage | Auto checkin | Vitals |
| Doctor Consultation | — | Presenting Complaints, History, Notes, ICD-11, Prescribe, Order New Investigation, Documents |
| Laboratory | Is a lab queue | Ordered, Approved and Published Investigations, Machine Results |
| Pharmacy | Is a pharmacy queue; connected store | Ordered Drugs, Dispensed |
| Radiology | Is a radiology queue | Ordered and Published Investigations |

### 3.4 Users, roles and permissions

Menu: `Users & Staff`. Section permission: `users.manage`.

Set up in this order: categories and specialities, roles, then users.

#### Staff categories and specialities

- **Categories** (`/users/categories`): groups of staff such as Doctor, Nurse, Support Staff. Click **Add Category**.
- **Specialities** (`/users/specialities`): clinical specialities such as Paediatrics. Click **Add Speciality**. A speciality can also be added while creating a user.

#### Roles

Menu: `Users & Staff > User Roles`. Page address: `/users/roles`. Permission: `users.roles.view`.

1. Click **Add Role**. Enter the **Role Name** * (for example Receptionist, Cashier, Doctor) and an optional description.
2. Open the role. The permission screen groups permissions by module. Each module shows a grid of **View, Create, Update, Delete** for its records, plus a **Special** list for permissions that are not simple record actions (for example approving a waiver).
3. Tick the permissions and save. "Permissions saved successfully" confirms it.
4. Set **Concurrent multi-facility viewing**:
   - On: users with this role see data from every facility they are assigned to at the same time.
   - Off: users with this role see only the facility they currently have selected.

If the permission screen says "No permissions found. Ensure the permissions seeder has run", the organization's permission list was not loaded when it was created. This is fixed by support on the server, not from the screen.

Section 21 lists the permission codes and suggests which ones each job role needs.

#### Create a user

Menu: `Users & Staff > Create New`. Page address: `/users/create`. Permission: `users.users.create`.

1. **Registration Type:** choose **Practitioner (Doctor/Clinician)** or **Non-Practitioner (Other Staff)**. The rest of the form appears after you choose.
2. **User Profile Details:** Title, Gender, Date of Birth, First Name, Middle Name, Last Name, ID Number, Phone Number.
3. **Authentication Details:**
   - **Username** — leave blank to auto-generate. If the name is in use the form suggests another ("Username taken. Try: …").
   - **Email**
   - **Role(s)** — optional here; can be added later.
4. **Employment Details:**
   - **Regulator** — KMPDC, NCK, PPB, COC, KNDI or KMLTTB — and the registration number
   - **Practitioner ID (PUID)** — the DHA Health Worker Registry id. It is filled in by the registry search and is needed for the Shared Health Record
   - **License No.** and **License Expiry Date** (an expired licence is flagged)
   - **Speciality**, **Employee Category**, **Department**, **Check-Roll Number**, **Job Description**
5. Save. The **User Created** box shows the login details once. Copy them and give them to the user.

#### Manage an existing user

Menu: `Users & Staff > View Existing` (`/users`). Find the user, then choose **Edit** or **Profile**. The profile has these tabs:

| Tab | What it does |
|---|---|
| Profile Information | Personal and employment details, employee category, department |
| Signatures | **Upload** an image or **Draw** the user's digital signature. It prints on signed documents such as notes, clearances and consent forms |
| Assigned Roles | Select the roles and click **Update roles** |
| Special Permissions | Grant individual permissions on top of the roles, using the same grid as roles |
| Authorized Regions & Facilities | Tick the facilities the user can work in, then **Save Regions & Facilities** |
| Authorized POS Stations | Tick the points of sale the user may open, then **Save POS Stations** |
| System Access | **Can Login**, dashboard and module access switches, login history, and **Force Logout** |
| Credentials | Change username, email and password; force a password reset |

Every clinician who signs documents should have a signature on file. Every user needs at least one facility, or they will see no data.

#### Why can't a user see a menu or button?

Check in this order:
1. The user has a role, and the role includes the permission for that screen (section 21).
2. The user is assigned to the facility under **Authorized Regions & Facilities**.
3. The sidebar section is switched on for the organization (section 3.8).
4. For a POS station, the user is authorised for that station.
5. For an outpatient queue, the destination is active and belongs to the user's facility.

After changing roles or permissions, the user signs out and signs in again.

### 3.5 Procedures, services and prices

Everything that is billed except stock items is a **procedure**: a consultation, a lab test, an imaging study, a minor procedure, a ward round. Set up categories first, then the procedures, then their prices per facility.

#### Procedure categories

Menu: `Setup > Procedures Settings > Procedure Categories`. Page address: `/evaluation/setup/procedure-categories`. Permission: `settings.edit`.

Click **Add Category**. Fields: **Name**, and **Applies To** — which clinical module the procedures belong to. Tick "A non clinical category" for services that are not clinical; all procedures in such a category appear under the "other services" point of sale.

Many categories can be loaded at once with **Bulk Import Categories (CSV)**: click **Download Template**, fill it in, choose the **CSV file** and click **Import**.

#### Add a procedure or service

Menu: `Setup > Procedures Settings > Add Service`. Page address: `/evaluation/setup/procedures/create`.

**General details**
- **Procedure name** (for example "Full Blood Count")
- **Procedure code** (for example FBC001, up to 50 characters; auto-generated if left blank)
- **Category**
- **Gender** (Any, Male or Female), **Sub title**, **Brief description**
- **Use biological reference flagging** — results are checked against reference ranges
- **Has quick results**

**Laboratory details** (for lab tests)
- **Parent procedure** — makes this a sub-test of a panel, for example Haemoglobin under Full Blood Count
- **Lab categories**
- **Result Type** — Text, Number, Drop Down, Rich Text, Range or Other. For Drop Down, type each option and click +
- **Default result template** — pre-fills the editor when results are recorded
- **Unit of measure**, **Sample Type**, **Lab additive**, **Turn around time**
- **Independent order** — the test can be ordered on its own at the lab
- **Multiple orders** — the test can be ordered more than once in a visit

**Consumables** — tick **Procedure consumes items from inventory** and add the products and quantities. They are deducted from stock automatically each time the procedure is performed.

**Tags** — add tags, including a revenue tag, used to group services in reports.

Click **Save Procedure**.

#### Set prices per facility

Menu: `Setup > Procedures Settings > View Services`. Page address: `/evaluation/setup/procedures`. Permission: `procedures.view`.

On a procedure choose **Update Prices**. For each facility set:

| Field | Meaning |
|---|---|
| **Cash Charge** | Price for cash patients |
| **Insurance Charge** | The general price for insured patients (see section 3.6 for scheme and company prices) |
| **Charge procedure at reception?** | The service can be charged at check-in as a pre-paid fee |
| **Has variable pricing (thus editable)** | Staff may change the price when billing |
| **Requires pre-auth** | Pre-authorisation is needed, with finer options: when using cash, when using insurance, only for specific insurance companies, or only for specific schemes |

Other actions on the list: **View All Prices**, **Template** (lay out the sub-tests of a panel, with optional titles and bold lines), and **Suggest ICHI / LOINC codes** to attach standard codes. A price list can be uploaded in bulk; rows not found in the catalogue can be created as new services or skipped.

A procedure with no price cannot be billed unless the setting that allows ordering procedures without a price is on (section 20).

#### Packages

Page address: `/core/setup/packages`. A package is a named bundle with a fixed **Amount**, a code and an Active switch.

### 3.6 Insurance companies, schemes, exclusions and prices

Menu: `Setup > Insurance/ Corporate`. Permission: `insurance.view`.

#### Insurance companies

Page address: `/core/setup/insurance-companies`. Click **Add Insurance Company**.

Required: **Company Name** *, **Postal Address** *, **City/Town** *, **Telephone** *, **Email Address** *. Optional: Postal Code, Mobile Number, Procuring Officer Name, Building, **KRA Pin**, **Manager**, Status (Active, Inactive, Suspended), Notes.

Tick **Is Corporate** for a company that pays for its staff directly rather than an insurer.

From a company's row you can open **Schemes** and **Drugs & Procedure Pricing**, and **Deactivate** / **Activate** it.

#### Schemes

Page address: `/core/setup/insurance-schemes`, or the **Schemes** button on a company. Click **Add Insurance Scheme** / **Add New Scheme**.

| Field | Notes |
|---|---|
| **Company** * | The insurer the scheme belongs to |
| **Scheme name** * | For example "Corporate Plan" |
| Scheme Code | Short code |
| **Scheme Type** * | Full, Capitation, Copay, Capitation with Copay, Average Cost, or Average Cost with Copay |
| Category | Individual, Corporate, Government, SHA or Private |
| Amount / Capitation Amount | The capitation amount for capitation types. Not applicable for a Full scheme |
| Copay Amount, Copay % | What the patient pays per visit |
| Max Coverage | The cover limit |
| Attention | The billing contact at the insurer |
| Effective Date, Expiry Date | The period the scheme is valid |
| **Smart Enabled** | The scheme uses a Smart card |
| **Policy No. Compulsory** | A policy number must be captured for patients on this scheme |
| **Requires Pre-auth** | Pre-authorisation is needed |
| **Age Restricted** / **Gender Restricted** | Limit the scheme to an age range (for example "18-65") or a gender |
| Status | Active, Inactive, Suspended |

For SHA, create the Social Health Authority as a company and its scheme with category **SHA**, then point the SHA settings at that scheme (section 3.21).

#### Scheme exclusions and scheme prices

On a scheme open **Drugs & Procedure Exclusions**. Page address: `/core/setup/insurance-schemes/{id}/exclusions-and-pricing`. There are separate lists for **Procedures** and **Products**. For each item you can:

- **Exclude** it (the scheme does not cover it) or **Include** it again
- Set a **Scheme Price** that overrides the general insurance price
- Tick **Requires pre-auth**

The list shows the **Cash Charge**, **Insurance Charge** and the scheme price side by side. "Not set" means no scheme price exists.

**Upload Excel** loads exclusions and prices in bulk. Keep the sheet in the format shown in the dialog. After the upload a summary shows the rows applied and the rows skipped.

#### Company prices

On a company open **Drugs & Procedure Pricing** (`/core/setup/insurance-companies/{id}/pricing`). A company price is the fallback used by every scheme of that company that has no price of its own. Set a company price for a procedure before ticking **Requires pre-auth** on it.

#### How the price for an insured patient is chosen

For a patient checked in under an insurance scheme, the system uses the first price that exists, in this order:

1. The **scheme price** for the item
2. The **company price** for the item
3. The item's general **insurance price** (the Insurance Charge on a procedure, or the Insurance Price on a product)

Points to remember when setting prices:
- A blank price is not the same as zero. Leave a price blank when there is none; enter 0 only when the item really is free.
- If none of the three insurance prices exists, the cash price is used as a last resort. Billing an insurer at the cash rate is usually a mistake, so fill in the insurance price.
- Cash patients, and insured patients with no scheme on the visit, are charged the cash price.
- An item marked **Excluded** under the scheme has to be paid in cash.

### 3.7 Payment modes, banks, service charges and rebates

**Payment modes** — `/settings/payment-modes`. Permission: `settings.edit`. Fields: **Name** * (for example Cash, M-Pesa, Credit Card), Code (for example cash, mpesa, card), Description, Active. Which modes the cashier is offered is also controlled by finance settings (section 20): cash, card, M-Pesa integrated and non-integrated, Pesapal card and M-Pesa, wallet.

**Banks** — `Setup > Finance > Banks`, `/finance/settings/banks`. Permission: `finance.view_financials`. Fields: **Bank Name** *, Description. Used for cheque and bank payments.

**Service charges** — `Setup > Finance > Service Charges`, `/finance/service-charges`. Click **Add Charge**. Fields: Service Code, Service Name, **Service Type** (Consultation, Procedure, Investigation, Lab, Imaging, Radiology, Pharmacy, Other), Standard Price, Min Price, Max Price, Cost Price, Effective Date, Active, Taxable and Tax %. Under **Recurrence Settings** choose **One-off charge**, or a recurring charge with **Cycle Recurrence** (Daily, Weekly, Monthly, Annually), **Total Cycles** and **Floor charge on incomplete cycles**. Service charges are what ward and dialysis charges are built on.

**Rebates** — `Setup > HIS Client Settings > Rebates`, `/core/setup/rebates`. Fields: **Name** *, Code, **Type** * (Percentage or Fixed amount), the amount or percentage, Description, Active. Rebates are applied on inpatient charge sheets.

### 3.8 General Settings and sidebar sections

Menu: `Setup > General Settings`. Page address: `/settings/settings/core`. Permission: any of `settings.manage`, `settings.view`, `settings.edit`; saving needs `settings.edit` or `settings.manage`.

General Settings is organised by module. The list on the left includes Registration, Core, Dialysis, Evaluation, Finance, Inpatient, Inventory, POS, Psychiatry, Reception, Settings, SHA, SMS, Integrations, Theatre and Users. Each module shows its settings as **Toggles**, **Colors** and **Configuration** values, with **Save Changes** and **Reset**.

#### Most settings are locked on screen

Most settings are shown with the message "These settings are locked and cannot be changed here." This is deliberate: a single switch can change clinical or billing behaviour, so changes are made by Afya Analytics support on request. The settings an administrator can change on this screen are:

| Setting | What it does |
|---|---|
| Bypass prescriptions billing | Lets patients receive drugs without going through billing first (the pharmacy's emergency override) |
| Bypass investigations billing | Lets patients receive lab and radiology services without going through billing first |
| SHA integration | Switches the SHA module on for the organization |
| SHA default intervention code | The default SHA intervention code |
| SHA default scheme | The insurance scheme that represents SHA |
| SHA agent ID | The National ID of the facility's HealthID biometrics agent |
| SHA preauth notification email | The inbox that receives preauthorisation status updates. SHA rejects a preauth without it |

The SHA test-mode and sandbox settings are shown blurred and can never be edited on screen.

To change any other setting listed in section 20, raise a request with Afya Analytics support and name the setting and the value wanted.

#### Switching sidebar sections on and off

In `General Settings > Core`, the "Side menu" group has one switch per sidebar section: Point of Sale, SMS, Reception & Registration, OutPatient, In-Patient, MCH, Theatre, Oncology, Psychiatry & Rehab, Morgue, Renal/Dialysis, Inventory & Procurement, Finance, SHA and Reports.

- All are on by default except **Morgue**.
- Switch off the modules a client does not use, so staff see a shorter menu.
- Dashboard, Users & Staff and Setup are not in the list and are always shown.
- Switching a section off also greys out its setup screens and its module on the General Settings page.

#### Registration settings (the patient form)

`General Settings > Registration` controls the patient registration form. For each field choose **Visible** and **Mandatory**. Ticking Mandatory also makes the field visible; unticking Visible also removes Mandatory.

- **Patient profile information:** First Name, Middle Name, Last Name, Date of Birth, Gender, Marital Status, Ethnicity, ID Number, TSC Number, Nemis Number
- **Patient contact information:** Mobile, Secondary Number, Email Address, Secondary Email Address, Telephone, Nationality, County of residence, Sub county of residence, Resident Village, Town, Home, Postal code, Postal address, Other details
- **Chief's information:** Chief first name, middle name, last name, area and mobile number

Click **Save Registration Settings**. Decide these with the client's records officer before registration starts, because changing mandatory fields later leaves older records incomplete.

#### Integrations settings

`General Settings > Integrations` lists the external connections available to the organization, each with an on/off switch, and where applicable a switch per individual API. Statuses include "Not connected" and "Turned off". Users without the right to change them see "You can view these switches but not change them."

### 3.9 Other setup screens

| Screen | Menu / page address | Fields and use |
|---|---|---|
| Partner Institutions | `Setup > Partner Institutions`, `/core/setup/partner-institutions` | Name *, Code, Type, Status, City, Phone, Email. Other hospitals, clinics and referral institutions |
| Appointment Categories | `Setup > Appointment Categories`, `/reception/setup/appointment-categories` | Category Name * (for example Consultation, Follow-up, Emergency), Description, Active. Offered when booking an appointment |
| Integrations | `/core/setup/integrations` | External systems (HIS, EMR, LIS, PACS, Other). See 3.23 |
| System Modules | `/core/setup/modules` | The backend services and their health. See 3.24 |
| Audit Trail | `Setup > Sudo > Audit Trail`, `/core/setup/audit-trail` | Who created, updated or deleted what, with **View changes**. Filter by date, event, facility and description. Permission: `audit.view` |

### 3.10 Check-in behaviour you can configure

These behaviours at reception are controlled by settings (section 20). They are set by support on request.

| Behaviour | Setting key |
|---|---|
| Skip the consent step at registration | `reception.skip_registration_consent` |
| Hide Middle Name, Town, Home Location, Occupation, County, Sub-county or Village on the patient form | `reception.remove_middle_name`, `reception.remove_town`, `reception.remove_home_location`, `reception.remove_occupation`, `reception.remove_county_of_residence`, `reception.remove_sub_county`, `reception.remove_village` |
| Limit how many dependants a patient can have | `reception.number_of_dependants` |
| Make choosing a clinician compulsory at check-in | `reception.make_clinicians_checking_mandatory` |
| Make the performing doctor compulsory for pre-paid procedures | `reception.performing_doctor_compulsory` |
| Require pre-paid fees at check-in | `reception.enforce_prepaid_fees` |
| Limit appointments per hour | `reception.appointments_per_hour` |
| Limit which roles patients can be checked in to | `reception.checkin_to_roles` |

### 3.11 Branding: colours and themes

Menu: `Setup > UI Colors`. Page address: `/settings/settings/settings`. Permission: `settings.edit`.

**Color Settings** lets you pick the **Navbar Color**, **Sidebar Color** and **Login Page Color** by colour picker or hex value, with a **Live Preview** and a **Login Page Preview**. **Save Changes** applies them; **Reset Defaults** returns to the standard colours; **Discard** drops unsaved changes. "Enter a valid hex color" means the value is not a valid colour code such as `#1E88E5`.

Together with the organization's logo and letterhead (section 3.2), this gives each client its own look on the sign-in page, in the application and on printouts. Each user can also switch between light and dark mode from the top bar.

### 3.12 Inventory setup

Menu: `Inventory & Procurement`, and `Setup > Inventory Settings`. Section permission: `inventory.manage`.

Set up in this order.

**1. Tax categories** — `/inventory/tax-categories`. Name, type and rate. Applied to products.

**2. Units of measure** — `/inventory/units-of-measure`. Unit name and description, for example Tablet, Bottle, Vial.

**3. Product categories** — `/inventory/product-categories`. Click **Add Category**. A category has a name, an optional parent category, a kind (**Consumable**, **Drug** or **Optical**) and default pricing percentages for **Cash (%)** and **Credit (%)**.

**4. Brands** — `/inventory/brands`. Name, code and status.

**5. Payment terms** — `/inventory/payment-terms`. Term name, number of days and status. Used on purchase orders.

**6. Products** — `Products > Product Management`, `/inventory/products`, **New Product**. Permission: `inventory.items.view`.

| Field | Notes |
|---|---|
| **Generic Name** * | The product's own (drug or generic) name |
| Product Code | Your stock code |
| **Category** * | From step 3 |
| DHA product catalogue | Links the product to the national standard drug concept so prescriptions are shared under the standard name. Use **Look up DHA codes** on the product list to get suggestions in bulk |
| Level of Use | Level 1 to Level 6 |
| **Unit of Measurement** * | The unit the product is sold in |
| Brand, Tax Groups, Formulation, Label Type | Classification |
| Cost | What you pay for one sold unit |
| **Selling Price** | The price of ONE sold unit (the unit of measurement), not of the big pack |
| Insurance Price | The general insurance price for the product |
| Discount | A standing discount |
| **Reorder Level** | The quantity at which the product counts as low stock |
| **Dispensable Amount** | Tablets or ml inside ONE sold unit. Prescriptions round up to whole units |
| Strength | For example 500mg |
| **Pack Size** | Sold units in one purchase pack |
| Sold As | **W — whole pack** or **P — per piece**: whether a pack may be split when dispensing |
| Product Group | Controlled and Chronic products require a prescription; OTC does not |
| **Requires Prescription** | The product cannot be sold over the counter without one |
| **Track Expiry**, **Track Batch**, **Track Serial Numbers** | Tracking expiry implies tracking batches. Tracked products need these values on every stock line |
| Product is price inclusive | The price already includes tax |
| Product is active in inventory | Untick to retire a product |

The **Packaging helper** works out prices from a big box: enter the big box price, the small boxes per big box and the tablets per small box, choose what the patient is billed in, and the helper fills the selling price, dispensable amount and pack size.

Optional clinical fields (Indications, Dosage, Adverse Reactions, Interactions, Contraindications, Pregnancy Safety, therapeutic classification, Kenya Essential Medicines List references and AWaRe class) support prescribing.

Bulk tools on the product list: **Bulk Update Prices**, **Bulk Update Prices (CSV)** with **Download CSV (Selected)** and **Apply Updates**, and **Print Product Labels**.

If you switch on batch or expiry tracking for a product that already has stock, the system recommends selling or adjusting the existing stock to zero first (**Sell/Adjust Stock First**), or you can **Continue Anyway**.

**7. Stores** — `Stores > Create Store`, `/inventory/stores/create`. Permission: `inventory.stock.create`.

| Field | Notes |
|---|---|
| **Store Name** * | For example Main Store, Outpatient Pharmacy |
| Store Code, Description, Location | Identification |
| Facility, Department | Where the store belongs |
| Parent Store | The store this one is supplied from |
| **Store Type** * | Main, Pharmacy, General / Departmental Store, Ward, Theater, Laboratory, or Consumable (Kitchen, Cafeteria and similar) |
| **Main Store** | The facility's central store |
| **Can order from suppliers** | The store may raise purchase orders and receive from suppliers |
| **Can update product prices** | Staff at this store may change prices |
| **Automatically save drafts** | Unfinished documents are saved as drafts |
| **Consumable** | Stock issued to this store is posted as an expense |
| **Restrict opening hours** | With **Open Time** and **Close Time**. Outside these hours prescribers see "Store closed" |
| Active | Untick to close the store |

A typical arrangement is one Main store that orders from suppliers, plus a Pharmacy store per dispensing point, and Ward, Theatre and Laboratory stores that receive stock from the Main store.

**8. Suppliers** — `Suppliers > Add New Supplier`, `/inventory/suppliers/create`. Permission: `inventory.suppliers.create`. Fields: **Supplier Name** *, Email Address, Phone Number, Mobile Number, Fax Number, Postal Address, Postal Code, City/Town, Street, Building, Active.

**9. Approval levels** — `Setup > Inventory Settings > Approval Levels`, `/inventory/setup/approval-levels`. Each level has a **Weight** *, the **users** assigned to it and a description. Levels are ranked by weight: a higher weight means more authority. Requisitions and purchase orders pass through these levels. Whether a user may approve their own order is a setting (`inventory.can_approve_own_order`).

**10. Opening stock** — for each store, open `Stores > List Stores > Manage > Add Items`. Search for each product and enter the **Quantity**, **Unit Cost**, and for tracked products the **Batch number**, **Expiry Date** and **Serial Numbers**, then click **Add to Store**. This records the opening balance as a stock movement.

`Setup > Inventory Settings` also has **Stores Setup** and **Suppliers Setup**, which edit, activate and deactivate stores and suppliers.

### 3.13 Point of sale setup

Menu: `Setup > Pos setup`. Page address: `/settings/pos/management`. Permission: `pos.terminals.manage`.

Create one point of sale for each till or selling point.

| Field | Notes |
|---|---|
| **Display Name** * | The name staff see in the sidebar |
| Patient type | The kind of customer the till serves |
| **Pos type** * | The kind of point of sale |
| **Store** * | The inventory store the till sells from |
| Service types/categories | The categories of items sold at this till |
| Icon | The sidebar icon |
| **Facility** * | Where the till is |
| **Is it a pharmaceuticals pos** | Turns on the pharmacy workflow: dosage, frequency and duration on each line |
| **Is it a custom pos** | A till for non-stock services |
| **Has stocks involvement** | Sales reduce stock in the store |
| **Allow invoicing and pay later to staff members** | Staff can buy on account |
| **Take payment at this till (Save and receive payment)** | The cashier at this till takes payment. If unticked, the sale is sent to the Cash Workbench to be paid there |
| Active | Untick to retire the till |

POS configurations (`/core/setup/pos-configs`) hold further till options: **Opening Float (KES)**, **Shift Start Time** and **Shift End Time**, **Max Discount %**, **Allowed Payment Methods**, **Auto Print Receipt**, **Require Shift Reconciliation**, **Enable Discounts** and **Manager Approval for Discounts**. POS devices (`/core/setup/pos-devices`) register the hardware: device name and ID, the configuration it uses, device type (Desktop Terminal, Tablet, Mobile, Dedicated POS), serial number, and attached printer, cash drawer, barcode scanner and card reader.

After creating a point of sale, authorise each cashier for it: open the user, then **Authorized POS Stations** (section 3.4). A till appears in a user's sidebar only when they are authorised for it and hold `pos.sales.create`.

Receipt layout is controlled by settings: paper size (80mm by default), header and footer text, whether to show the logo and the station (section 20).

### 3.14 Inpatient setup

Menu: `Setup > Inpatient`. Permission: `inpatient.manage_wards`.

Set up in this order.

1. **Bed Types** (`/inpatient/settings/bed-types`) — name and description, for example General, Private, ICU.
2. **Admission Types** (`/inpatient/settings/admission-types`) — name, **Deposit (Ksh)**, description and associated procedure. The deposit is the amount expected before admission.
3. **Discharge Types** (`/inpatient/settings/discharge-types`) — code, name and description.
4. **Wards** (`/inpatient/settings/wards`). Permission: `wards.view`.
   - Ward Number
   - **Ward Type** * — pick a standard type or enter your own
   - Display Name — defaults to the ward type followed by "Ward"
   - **Provides maternity care** — adds the Partograph, Report on Labour and Labour Care Guide tabs to the chart
   - **Psychiatry Service** — Mental Health or Rehabilitation. Leave blank for an ordinary ward. Admissions to a psychiatry ward go through the legal-form check
   - Gender (Male, Female, Other), Age Group (Adult, Children)
   - **Cash Cost** and **Insurance Cost** — the daily bed charge; select an existing charge or type an amount
   - Associated Procedure and Associated Procedure Category
   - Ward tabs are assigned automatically by ward type
5. **Beds** (`/inpatient/settings/beds`). Permission: `inpatient.beds.view`. **Bed Number** * (for example 32K), **Ward** *, Bed Type.
6. **Recurring & One-off Fees** (`/inpatient/settings/charges`). **Name** *, **Cost** *, **Type** * (One-Off Charge or Recurring Charge), the wards it applies to, associated procedure, and **Post automatically on admission** — the charge is ticked for every admission to those wards and can be unticked per patient.

Related settings: automatic IP numbers and their prefix, whether wards can be over-booked, and whether the charge sheet of a discharged patient can still be edited (section 20).

### 3.15 Theatre setup

Menu: `Setup > Theatre Settings`. Permission: `theatre.setup`.

1. **Theatre Types** (`/theatre/setup/theatre-types`, permission `theatre.types.index`) — name, description, associated procedures and status.
2. **Operation Theatres** (`/theatre/setup/theatre`, permission `theatre.theatre.index`) — each operating room: name, number, type, facility, associated procedure, **Charging Cycle**, **Theatre Fees** and status.
3. **Theatre Deposits** (`/theatre/setup/theatre-deposits`, permission `theatre.deposits.index`) — named deposit amounts per facility, selected when a booking is created.

Theatre behaviour set through settings (section 20): how theatre is charged (`theatre_payment_method`: inclusive, one-off or per-cycle), whether the nurse and doctor ward checks are required before a patient goes to theatre, and whether the ward-check printout is available.

Theatre is its own module: do not create it as a destination.

### 3.16 Dialysis setup

Menu: `Setup > Dialysis Settings`. Permission: `dialysis.setup`. Build in this order, because each level refers to the one above it.

1. **Dialysis Plants** (`/core/setup/dialysis-plants`) — the water-treatment plant: name, location, capacity, description. Each plant has a **Water Tests** log: conductivity, hardness, chlorine, pH, bacterial count, endotoxin and notes, with the result and who tested.
2. **Dialysis Ward Types** (`/core/setup/dialysis-ward-types`) — name and description.
3. **Dialysis Wards** (`/core/setup/dialysis-wards`) — ward name, number, **Ward Type** *, plant, capacity, and **Flat Rate (session package charge)**.
4. **Dialysis Machines** (`/core/setup/dialysis-machines`) — display name, machine number, make, model, serial number, plant, and status (Operational, Under Maintenance, Out of Service). Each machine has a maintenance log (Scheduled, Breakdown, Repair) with downtime in minutes.
5. **Dialysis Chairs** (`/core/setup/dialysis-chairs`) — chair name or number, its ward and its machine. The chair is what gets booked: schedules book chairs, not machines.
6. **Vascular Access Types** (`/core/setup/dialysis-vascular-accesses`) — for example AV Fistula, Central Line, Graft.
7. **Coagulation Types** (`/core/setup/dialysis-coagulation-types`) — the anticoagulation options.
8. **Dialysis Deposits** (`/core/setup/dialysis-deposits`, permission `dialysis.deposits.manage`) — named deposit or fee amounts.

Sessions cannot be scheduled until at least one active machine and chair exist. Dialysis is its own module: do not create it as a destination.

### 3.17 Morgue setup

1. Switch the **Morgue** sidebar section on in `General Settings > Core` (it is off by default).
2. Grant mortuary staff the morgue permissions (section 21).
3. Open `Morgue > Settings` (`/morgue/settings`) and click **Add a fridge**: **Name** (for example "Main fridge"), **Code**, **Kind** (Fridge, Freezer or Transit), **Target °C**, where it is, and **How many spaces** with a **Label prefix**. The spaces can be added or renamed later.

"No fridges yet. Add one before receiving a body" appears until a fridge exists. "You don't have permission to configure the morgue" means the user lacks the storage management permission.

### 3.18 Psychiatry setup

Menu: `Setup > Psychiatry Settings > Psychiatry`. Page address: `/core/setup/psychiatry`. Permission: `psychiatry.settings.manage`.

| Group | Settings |
|---|---|
| Which forms this clinic uses | A switch per form |
| Patient-facing wording | What the clinic is called in reminders and printouts (never the clinic's real name, because this goes out by SMS and on paper), the **Crisis line** printed on every aftercare plan, and the **Programme name** |
| Safety gates | **Block a psychiatric admission until its legal form is complete**; **Block discharge until the discharge summary is signed** (off by default: a missing summary is then only a warning); **Raise the triage colour when risk is recorded as high** (raises the visit to Very urgent and never lowers a colour another clinician set) |
| Timings | Hours of written notice before a voluntary patient may leave (statutory value 72); hours within which an emergency admission must arrive (statutory value 24); how long emergency access to a record lasts, in minutes; minutes before an unacknowledged urgent alert is escalated; reminders when a legal review falls due |

Also needed:
- Create the Psychiatry, Rehabilitation and Therapeutic destinations the client uses (section 3.3).
- Mark psychiatric wards with a **Psychiatry Service** (section 3.14).
- Grant each clinician the service permissions they need: `psychiatry.mental_health.access`, `psychiatry.rehabilitation.access`, `psychiatry.therapeutic.access`. The general clinical permission does not open psychiatric records.
- Give the clinical lead `psychiatry.break_glass.review` so emergency access is reviewed.

### 3.19 Oncology setup

Oncology has no setup screens of its own in the application. Screening programmes, intervals, treatment protocols and benefit rules come from the oncology registry on the server. To prepare a client:

- Keep the **Oncology** sidebar section on.
- Grant the permissions by role: `oncology.screening.record` for screening staff; `oncology.diagnosis.write`, `oncology.staging.write`, `oncology.mdt.decide` and `oncology.plan.approve` for oncologists; `oncology.cycle.open` and `oncology.cytotoxic.handle` for day-unit nurses and oncology pharmacists; `oncology.toxicity.write` and `oncology.benefit.view` as needed.
- Make sure the chemotherapy drugs exist as products with stock in the relevant store.

### 3.20 MCH setup

The MCH section (ANC, PNC, CWC and Family Planning) uses the general clinical permission `evaluation.manage`. There are no separate setup screens. Patients enter the enrollment queues from the triage vitals screen, so make sure the triage destination has the Vitals action. The registers (MOH 405, 510, 511 and 711) are produced from the data recorded in these clinics.

### 3.21 SHA (Social Health Authority) setup

Complete all of these for SHA claims to work:

1. **Switch SHA on:** `General Settings > SHA`, switch on SHA integration. The SHA sidebar section and SHA reports then appear.
2. **Create the SHA scheme:** add the Social Health Authority as an insurance company and its scheme with category **SHA** (section 3.6).
3. **Point the settings at it:** in `General Settings > SHA` set the **default scheme**, the **default intervention code**, the **agent ID** (the National ID of the facility's HealthID biometrics agent) and the **preauth notification email**.
4. **Facility code:** enter the **SHA FR Code** on every facility that will claim (section 3.2).
5. **Practitioners:** each clinician needs their **Practitioner ID (PUID)**, regulator and licence number on their user record (section 3.4).
6. **Products and procedures:** link products to the DHA catalogue and attach standard codes to procedures, so claim lines carry recognised codes.
7. **Workstation:** install the HealthID application and fingerprint reader where consent is captured (section 3.22).
8. **Permissions:** grant `insurance_claims.view`, `insurance_claims.submit` and `insurance_claims.resubmit` to the claims team.

In daily use, a patient must have the SHA scheme and member details on their record, be checked in under the SHA scheme with the SHA services chosen, and have an ICD-11 diagnosis recorded.

### 3.22 Workstation devices: fingerprint reader and HealthID

**HealthID** is the Windows application SHA requires on the computer that owns the fingerprint reader. The facility's biometrics agent signs in to it. It runs in the background on that computer and the HMIS page talks to it directly. If the page reports "Could not reach HealthID on this computer", check that HealthID is installed, running and signed in on that workstation.

**SecuGen fingerprint reader** (Hamster Plus and similar USB readers): the SecuGen local service that ships with the SecuGen software must be installed and running on each reception computer that captures fingerprints. Fingerprint images are processed on the workstation and do not leave it.

Both are per-computer installations. A reception desk without them can still register and check in patients; it cannot capture fingerprints or biometric consent.

### 3.23 Integrations with other systems

Page address: `/core/setup/integrations`. Click **Add Integration**.

| Tab | Fields |
|---|---|
| Basic | **Name** *, Code, **Type** * (HIS, EMR, LIS, PACS, Other), Provider, Version, Description, API URL |
| Authentication | Auth Type (None, Bearer, OAuth2, Custom), API Key, API Secret, Username, Password, headers and credentials |
| Sync & Advanced | Sync Direction (Push, Pull, Both), Sync Interval in minutes, Auto Sync Enabled, Timeout, Retry Attempts, field mapping and filters |
| Extra | Enable Logging, Enable Webhooks, Webhook URL and secret, Status (Active, Inactive, Testing, Error, Suspended), Notes |

On each integration: **Test** checks the connection, **Sync Now** runs a sync, and **Logs** shows what happened.

Laboratory analysers are connected separately under `Setup > Lab Settings > Machines` (section 18). Product synchronisation with Microsoft Dynamics, where the client uses it, is shown on the product list as **Dynamics Sync Status** with **Sync Now**.

### 3.24 System modules and service health

Page address: `/core/setup/modules`. This screen is for support staff. It lists each backend service as a module with its code, version, service address and health, and shows whether it is a core module. **Check Health** tests one service; **Check All Health** tests them all. A module that is not a core module can be disabled. "No service URL" means the service address has not been configured.

Use it when one area of the application is failing, to see whether that service is down.

### 3.25 Bulk onboarding of an organization's users

Page address: `/admin/business-setup`. A three-step wizard for support staff:

1. **Organization Setup** — Business Name (at least 2 characters), registration number and currency.
2. **Add Employees** — download the CSV template, fill in one row per person with `full_name`, `email`, `phone` and `role`, and upload it. The file is validated and errors are listed before anything is created.
3. **Complete** — shows the users created and any that failed, with the reason. Created users are invited by email and activate their accounts through the setup link (section 2.2).

### 3.26 Go-live checklist

Before a client starts using the system, confirm each of these with a test patient:

- [ ] A receptionist can sign in, register a patient and check them in to Triage
- [ ] The triage nurse sees the patient in the queue and can record vitals
- [ ] The doctor sees the patient, writes notes, orders a lab test and prescribes a drug
- [ ] The cashier sees the bill in the Cash Workbench, takes payment and prints a receipt with the correct logo and details
- [ ] The laboratory sees the test, collects a sample, enters and publishes a result
- [ ] The pharmacy sees the prescription, dispenses it, and stock in the store goes down
- [ ] An insured patient is priced at the scheme price and appears in the Insurance Workbench
- [ ] If inpatient is used: a patient can be admitted to a bed, charged daily and discharged
- [ ] Each role sees only its own menus
- [ ] Printouts (receipt, invoice, prescription, lab report) show the organization's details
- [ ] The dashboard figures match what was done

---

## 4. Dashboard

Menu: `Dashboard`. Page address: `/dashboard`. Permission: `dashboard.view`.

The dashboard is the home screen after sign-in. It shows only the widgets your role allows:

- **Today at a Glance:** Today's Visits, All-Time Visits, Queue Waiting, Active Admissions, Awaiting Discharge, Theatre Today.
- **Revenue:** Revenue Today, This Week, This Month, This Year, plus Exemptions and Waivers This Week.
- **Clinical Operations:** Today's Theatre Schedule and ward occupancy.
- **Inventory Alerts:** Low Stock, Out of Stock, Expiring within 30 days, Expired.

Click **Refresh** to reload the figures. Each widget shows the same live data as its detail screen. If a widget and its detail screen disagree, report it to support.

If you see "No dashboard widgets are available for your role", ask your administrator to grant the dashboard permissions.

---

## 5. Reception and registration

Menu: `Reception & Registration > Reception`. Permission: `reception.manage`.

### 5.1 Register a new patient

Page address: `/reception/patients/create`. Open `Reception > Manage Patients` and start a new patient.

The form has these parts:

1. **Patient profile information:** First Name, Middle Name, Last Name, Date of Birth, Gender, Marital status, ID Type, ID Number.
2. **Patient contact information:** Mobile, Secondary Number, Email Address, Town, Home Location, Occupation, County of residence, Sub county of residence, Resident Village/Estate.
3. **Next of kin:** first, middle and last name, mobile, email and **Relationship**. If the next of kin is already a patient, search and select them.
4. **Dependants:** link an existing patient as a dependant, or use **Register a newborn under this patient**.
5. **Finance & insurance details:** tick **Patient has insurance**, choose the **Insurance Company** and **Insurance Schemes**, and enter the **Member Number**, **Policy Number** and **Principal**. Use **Add Scheme** for a second cover.

Save with **Save Patient**, or **Save and Checkin** to go straight to check-in.

Rules to know:
- Age is calculated from the date of birth.
- Patients aged 18 and over need an ID number. Patients under 18 show a **Minor** badge and the ID number is not required; a guardian's phone number is still useful.
- **ID Type** options: National ID, Birth Certificate, Birth Notification, Alien ID, Refugee ID, Mandate Number, SHA Client Registry ID, Passport.
- Which fields are shown or mandatory is controlled in General Settings (see 3.8).

### 5.2 Register a newborn

In the mother's (or guardian's) patient form, under dependants, click **Register a newborn under this patient**. Enter **First name** ("Baby" is fine until the baby is named), **Surname**, **Sex** and **Date of birth**, then click **Register and link**. The baby gets its own patient record, linked to the parent.

### 5.3 Capture patient consent

On the patient form, the **Patient Consent Form** opens a signature box. The patient signs with a mouse or on a touchscreen. Use **Clear signature** to start again. Once captured, the form shows the saved signature and date; use **Re-capture Signature** to replace it, or **Skip (Patient Refused)** if the patient declines.

### 5.4 Find and edit a patient

Menu: `Reception > Manage Patients`. Page address: `/reception/patients`. Permission: `reception.patients.view`.

Search by name, patient number or mobile. Each row has **Edit**, **Check In** and **Barcode**. The patient view shows **Details, Vitals, Visits, Prescriptions, Investigations, Procedures, Doctor Notes** and, under **Finance**, **Overview, Deposits, Refunds, Cash Payments, Invoice Payments, Waivers**.

### 5.5 Check a patient in (start a visit)

From `Manage Patients`, click **Check In** on the patient. Page address: `/reception/patients/{id}/visits/create`.

1. Under **Queueing details**, choose one or more **check in destinations** (for example Triage and Laboratory).
2. Choose the **Purpose of visit**.
3. Optionally **Specify doctor / Consultant**.
4. Choose the **Payment method**: **Cash / Mpesa / Card** or **Insurance**. For insurance, pick the **Insurance Scheme**.
5. For SHA patients, choose the **SHA services (intervention codes)**.
6. Optionally add **Pre-paid fees** by choosing procedures and the performing doctor.
7. Optionally record a **Referring Facility**.
8. Click **CHECKIN**, or **CHECKIN AND TREAT** to open the clinical screen straight away.

The page also lists the patient's **Last Visits**.

**"Patient already has an active visit"** — the patient was not checked out of an earlier visit. Choose **Open full visit** to go to it, **Check out** to close it, or **Create new visit**.

### 5.6 Manage checked-in patients

Menu: `Reception > Manage Queue`. Page address: `/reception/visits`. Permission: `reception.visits.view`.

Filter by start and end date and time, or search by patient. Each row offers:

| Action | What it does |
|---|---|
| **Update Visit** | Change the visit details, such as payment method or scheme |
| **Print Ticket** | Print the patient's queue ticket |
| **Change Destination** | Move the patient to a different queue, or add another queue |
| **Cancel Checkin** | Cancel the visit |

**Change Destination:** the dialog shows **Current Destinations**. Pick a new one and click **Change destination**, or use **Add Destination** / **Checkin elsewhere** to add a queue without removing the current one. If the patient is in more than one queue, first pick which one to change.

Opening a visit shows the **Patient journey** (checked in, vitals taken, seen by doctor), the destinations, finance totals (**Charged**, **Paid / covered**, **Outstanding**), insurance invoices, and tabs for **Vitals, Notes, Tests, Pharmacy** and **Consumables**.

### 5.7 Appointments

Menu: `Reception > Appointments`. Page address: `/reception/appointments`. Permission: `reception.appointments.view`.

To book: search for the patient (or tick **External Patient** for someone not yet registered — only when you are sure they are not already in the system), choose the **Facility**, the **Doctor / Department / Clinic**, the **Category**, **Date**, **Time**, **Priority** (Normal, Urgent, Emergency) and add **Instructions**.

Appointment statuses: Scheduled, Confirmed, Checked in, In progress, Completed, Cancelled, No show.

The list can be filtered by date, patient, clinician, department, clinic and status, and printed with **PRINT**. `Reception > Calendar` shows appointments in a calendar view.

### 5.8 Follow-ups

Menu: `Reception > Patients Follow up`. Page address: `/reception/follow-up`.

This list is filled automatically from the **Next steps** a doctor records in their notes. Filter by **Pending**, **Completed** or **All**, by clinician and by date. Use **Book appointment** to turn a follow-up into an appointment, and **DONE** to mark it completed.

### 5.9 Patient documents

Menu: `Reception > Patient Docs`. Page address: `/reception/patient-docs`. Search for the patient and click **VIEW FILES** to see or add scanned documents.

---

## 6. Outpatient: queues and clinical work

Menu: `OutPatient`. A user sees this section with any of: `evaluation.manage`, `lab.tests.view`, `pharmacy.dispensing.view`, `triage.view`, `triage.perform`.

### 6.1 How queues work

The OutPatient section has one entry per destination set up for your facility (see 3.3). A checked-in patient appears in every queue their visit was sent to. A patient sent to Consultation, Laboratory and Pharmacy appears in all three.

Each queue lists **Name, Date / Time, Facility, Destination, Vitals recorded, Seen by doctor?, Has results?** and an **Actions** column. Filter by date and time, facility or patient name; use **Refresh** and **Reset**.

Row actions depend on the queue:
- **MANAGE** opens the patient's clinical screen.
- **SAMPLE COLLECTION** (laboratory queues) opens sample collection.
- **PROCESS MEDS** (pharmacy queues) opens dispensing.
- **CHECKOUT** removes the patient from the queue.

### 6.2 Triage colours in the queue

Rows are coloured by urgency:

| Colour | Meaning |
|---|---|
| Green | Routine — not urgent |
| Yellow | Urgent |
| Orange | Very urgent |
| Red | Emergency — resuscitation |

The colour is set from the vitals recorded at triage.

### 6.3 The patient clinical screen

Clicking **MANAGE** opens the visit. The header shows the patient's details and warning chips for **Food allergies**, **Drug allergies**, **Other allergies** and **Chronic illnesses** ("NKDA" means No Known Drug Allergies; "NKFA" means No Known Food Allergies; "not yet assessed" means nobody has recorded it).

Header buttons:
- **VITALS** and **PROCEDURES** — quick access to those records
- **CHECKIN ELSEWHERE** — send the patient to another queue
- **CHECKOUT** — end the patient's time in this queue
- **Request Admission** — ask for the patient to be admitted (see 8.1)

The tabs below the header are the **Treatment Actions** configured for that destination, so different queues show different tabs.

### 6.4 Triage and vitals

Tab: **Vitals**. Click **Add New Vitals** and record Weight, Height (BMI is calculated), BP Systolic and Diastolic, Pulse, Respiration, Temperature and location, Oxygen saturation, blood sugar (Random or Fasting, in mmol/L or mg/dL), and for children head circumference and M.U.A.C. Add **Nurse's notes** and **Current Medication(s)**.

- Entries save as you go. Fields marked with * are required to finish triage when your facility requires core vitals.
- The system flags values outside the normal range for the patient's age.
- When finished the screen says "Triage complete. The patient can proceed to the selected destination."
- For pregnant patients record the **Last Menstrual Period (LMP)**; the **Expected Date of Delivery (EDD)**, gestation and trimester are calculated, and **Enroll to ANC** sends the patient to the antenatal enrollment queue. **Enroll to CWC Clinic** and postnatal enrollment work the same way.

### 6.5 Doctor's notes

Tab: **Notes**. The doctor records **Presenting complaints** (symptom, duration and onset), history, examination, diagnosis and plan. Notes include the earlier notes for the patient under **All previous notes**.

**Diagnosis coding:** conditions are picked from the ICD-11 list. At least one ICD-11 condition is required for SHA claims, and sided conditions need **Left**, **Right** or **Bilateral**.

**Next steps** in the notes drive the reception follow-up list: **TCA** (come again after a number of days, weeks or months), **Refer to specialist**, **Get pre-authorization**, **Book for theatre**, **Call for review**, **Refer for investigation**.

The **Documents** tab produces a **Paid Sick Chit**, **Light Duty Chit**, **Referral Chit** or **Treatment and Back to Work Chit**, and stores uploaded files such as lab reports and discharge summaries.

Specialised clinics have their own tabs when configured for the destination, including eye (refraction, slit lamp exam, eyewear prescription), dental (odontogram, periodontal chart, dental clearance), ENT, orthopaedic and maternal examination forms.

### 6.6 Prescribing medicines

Tab: **Prescribe** (also **Create prescription** from the notes).

1. Choose the department or store, then search for the drug. The list shows real stock in that store.
2. Enter the **Dosage**, unit, **Route**, **Frequency** and **Duration**. The **Quantity** is worked out for you.
3. Optional ticks: **Drug substitution is allowed**, **Drug is for Admission**, **Drug is for Discharge**, **Drug is a priority**.
4. Add **Doctor's notes** and click **Save prescription**.

Warnings you may see:
- **Known Patient Allergies** or **Prior Adverse Drug Reactions** — read and click **Acknowledge** to continue.
- **Product not covered by patient scheme** — the patient pays cash for that item.
- **Requires pre-authorization** — the insurer must approve it first.
- **Store closed** — the store is outside its opening hours.

Use **Prescribe own medication** to record a medicine the patient already has and that the hospital is not dispensing.

### 6.7 Ordering laboratory tests and other investigations

From the notes or the investigations tab, order tests by searching for the procedure. Ordered tests show their status:

- **Procedure not paid for yet** — a cash patient must pay first
- **Awaiting Sample Collection**
- **Sample Received**
- Results entered, then approved and published

### 6.8 Laboratory: sample collection and results

**Collect a sample:** in a laboratory queue click **SAMPLE COLLECTION**. Click **New Sample**, choose the sample type and the procedures it is for, and save. From the list of **Collected Samples** you can **Print barcode**, **Queue to laboratory**, **Reject sample** (a reason is required) or **Recollect**.

**Enter results:** open the patient with **MANAGE**. Enter the result for each test. Results can be typed, picked from options, or uploaded as PDFs or images. For analysers connected to the system, **Machine Results** shows the readings received; the last result is used by default and you can choose **Use this** on another.

**Approve and publish:** results move from unapproved, to approved, to published. A doctor sees published results in the patient's visit, and the queue's **Has results?** column updates. Values outside the reference range are flagged, and critical values are highlighted.

### 6.9 Pharmacy: dispensing

In a pharmacy queue click **PROCESS MEDS**. Page address: `/evaluation/pharmacy/process-meds/{visitId}`.

1. **Pending Prescriptions** lists each drug with the prescribed quantity, what is already processed and the balance.
2. Tick the drugs and set the **Quantity to Process**.
3. Click **Process and Bill** to send the charge to the cashier, or **Process and Close**.
4. Once paid (or covered by insurance), open the ordered drugs and click **Dispense**. Use **Print Label** for the drug label.

Payment status shown on each drug: **Paid**, **Not paid**, **On insurance**, or **On admission bill** (inpatient drugs are dispensed now and settled on the admission bill).

Dispensing reduces stock in the store. **Return Drug** on a dispensed item records a return, with **Accept** or **Reject** by the pharmacy.

### 6.10 Checking a patient out

Click **CHECKOUT** on the queue row or in the patient header. Choose the **Discharge Type**, add notes if needed, and click **Confirm Checkout**. The patient leaves that queue. A patient who is never checked out will show "Patient already has an active visit" at their next check-in.

---

## 7. Finance and billing

Menu: `Finance`. Permission: `finance.manage`.

### 7.1 Overview of the finance screens

| Screen | Page address | Use |
|---|---|---|
| Cash Workbench | `/finance/cash/queue` | Patients with bills waiting to be paid in cash |
| Insurance Workbench | `/finance/invoices/pending` | Insurance and credit invoices from pending to dispatched |
| Insurance Management | `/finance/view/company-statements` | Statements and balances per insurance company |
| Cash Payments | `/finance/cash/evaluation` | All receipts issued |
| Sales Summary | `/finance/cash/sales-summary` | Collections per cashier and payment mode |
| Credit Notes | `/finance/credit-notes` | Raised credit notes and approvals |
| Patient Accounts | `/finance/accounts` | Each patient's deposits, balance and invoices |
| Patient Invoices | `/finance/cash/invoiced` | Cash invoices, balances and waivers |

### 7.2 Receive a cash payment

1. Open `Finance > Cash Workbench`. Permission: `finance.process_payments`.
2. Find the patient. Use the **Theatre** and **Morgue** toggles to see theatre and mortuary charges; there is no separate theatre billing screen.
3. Click **RECEIVE PAYMENT**. The unpaid bills for the visit are listed.
4. Tick the items being paid (or **Select all**), choose the payment mode (Cash, M-Pesa, Credit card, Cheque and others set up at your facility) and confirm.
5. Print the receipt as a **Thermal receipt (80mm)** or an **A4 page**.

"This patient has no outstanding bills. Payment cannot be processed." means everything is already paid or invoiced.

Other actions from the workbench: **PROCESS MEDS**, **CREATE INVOICE** and **VIEW ACCOUNT**.

### 7.3 Create a cash invoice (pay later)

From the Cash Workbench click **CREATE INVOICE**. Tick the items, optionally enter a **Discount amount** with a **Reason for discount**, and click **Proceed Creating Invoice for Selected Items**. The invoice then appears under `Finance > Patient Invoices`, where you can **Receive Payment**, **View Invoice**, add a **Waiver**, or **View Account**.

A 0 Ksh item cannot be billed. Set its price first.

### 7.4 Receipts, and reversing a payment with a credit note

`Finance > Cash Payments` lists every receipt. Filter by cashier, receipt number, patient, status, payment mode and date. Totals at the top show the amount collected by Cash, M-Pesa and card, and the amount cancelled. **Excel** exports the list.

To reverse all or part of a receipt:
1. Click **Issue Credit Note** on the receipt.
2. Tick the **Items to credit**, check the **Credit amount**, add **Notes** and **Save**.
3. The credit note is **pending approval**. A user with approval rights opens it and clicks **Approve Credit Note** (or **Cancel Credit Note**).

`Finance > Credit Notes` shows **All Credit Notes** and **Pending Approval**, with an activity log for each note. An approved credit amount can be moved to the patient's account with **Request Transfer**.

Crediting or cancelling a sale that included dispensed stock puts that stock back.

### 7.5 Waivers

Open the invoice in `Finance > Patient Invoices` and choose **Waiver**. Enter the **Amount** and **Remarks** and **Save**. The waiver waits for **Approve Waiver** by an authorised user. The page shows the invoice total, the amount paid, earlier waivers and the pending bill.

### 7.6 Patient accounts and deposits

`Finance > Patient Accounts`, then **View Account / Profile**. Permission: `finance.accounts.view`.

The account shows the **Account Balance** and tabs for deposits, refunds, payments, invoices and vouchers. Record a **Deposit** with the amount, payment mode and reference, then print the receipt in A4, A5 or thermal. Deposits are used for admission and theatre authorisation.

### 7.7 Insurance Workbench

`Finance > Insurance Workbench`. Permission: `finance.invoices.view`. The tabs follow the life of an insurance invoice:

| Tab | What is here | Main actions |
|---|---|---|
| **Pending** | Insured visits with items not yet invoiced | **Bill**, **Change**, **Process Meds**, **Split** |
| **Billed** | Invoices raised | Print (A4, A5, Thermal), **Split bill**, **Dispatch**, **Cancel** |
| **Cancelled** | Cancelled invoices | **Undo**, **Unbill Items** |
| **Dispatched** | Invoices sent to the insurer in batches | Print outs, **Download Excel**, **Email** |
| **Submitted to SHA** | Invoices submitted to SHA | View status |
| **SHA claim builds** | SHA claims being prepared (Queued, Building, Built, Held, Failed) | **Open claim** |

Filter by insurance company, scheme, date, queue (Outpatient, Inpatient, Theatre, Dialysis), patient and invoice number.

**To bill an insured visit:** on **Pending** click **Bill**. On **Billing Management**, tick the items and click **Bill Selected Items**. Other options: **Move this item to cash payment** (for items not covered), **Add Extra Items to Visit Bills**, **Change Payable Amount**, **Change Copay Amount**, **Split Patient Bill** and **Print Interim Invoice**. If the scheme has a capitation limit the page shows by how much the bill exceeds it.

**To dispatch invoices to an insurer:** on **Billed**, tick invoices from the same insurance scheme and click **Dispatch Selected Invoices**. They move to **Dispatched** with a batch number.

**To cancel an invoice:** choose **Cancel**, give the reason and confirm. It moves to **Cancelled**, where **Undo** restores it.

**Split a bill:** raises separate invoices to the patient and to one or more of the patient's insurance schemes for chosen items.

### 7.8 Insurance statements

`Finance > Insurance Management` shows the total, paid amount and balance per insurance company. Filter by company, scheme and date; export with **Excel**; **View** opens the company's billed invoices.

### 7.9 Sales summary and day close

- **Sales Summary** (`/finance/cash/sales-summary`): per cashier, the totals by Cash, M-Pesa, Card, Cheque, discounts, refunds, invoice sales, gross and net. Permission: `reports.view_financial_reports`.
- **Day Close** (`/finance/cash/day-close`): closes the day's sales. Click to close, then **Confirm**. This cannot be undone; you can close again the next day. **Print Report** prints the Day Close Financial Summary.

---

## 8. Inpatient

Menu: `In-Patient`. Permission: `inpatient.admissions.view`.

### 8.1 The admission process

1. **Request.** A clinician clicks **Request Admission** in the patient header (outpatient). They choose the **Admission Type**, the facility, a **Preferred Ward**, and record the reason, diagnosis, history, complaints and examination findings.
2. **Authorize.** `In-Patient > Awaiting Authorization` (`/inpatient/admission-authorization`). Click **Authorize**, confirm the **Ward** and the **Required Amount**, and enter the authorized amount — the deposit the patient must pay before a bed is released. Tick **Waive the deposit — admit without payment** if applicable. The authorisation is recorded against your name. Set the **Payment Mode** to Cash or Insurance Cover.
3. **Admit.** `In-Patient > Admission Requests` (`/inpatient/admission-requests`) or `Checkins/IPD`. Click **Admit**. Enter the **Admission Date**, the **Admission Doctor**, select a **Ward** and a **Bed**, and confirm the **Admission charge(s)**. The **IP Number** is auto-generated when your facility has that setting on.

Statuses on the authorization list: Not Authorized, Legal forms pending, Pending Deposit, Not ready to admit, Authorized.

A request can be cancelled from the list with a reason. Permission to admit: `inpatient.admit_patients`.

### 8.2 Managing admitted patients

`In-Patient > Patient Management` (`/inpatient/admissions`) lists all admitted patients with ward, bed, admission doctor and payment type. Row actions:

- **EVALUATE** — open the inpatient chart
- **EDIT ADM. DETAILS** — change ward, bed, payment method or admission date
- **CHANGE WARDS** — transfer the patient; pick the new ward and bed, the charges for the new ward and a **Reason for Transfer**
- **PRINTOUTS** — admission and charge printouts

### 8.3 The inpatient chart

**EVALUATE** opens the chart. The header shows IP number, account balance, deposit, admission date, days admitted, payment mode and admitting doctor. Tabs include:

Patient Vitals, Doctors (notes), Nursing Kardex, Treatment Sheet, Blood Pressure, Temperature, Investigations, Procedures, Blood Transfusion, Fluid Balance, Feeding, Nursing Care Plan, Consumables, Patient History, Inpatient History, Daily Charges, Daily Charge Sheet, Book To Theatre and Discharge.

Maternity wards also show **Partograph**, **Report on Labour** and **Labour Care Guide**. Psychiatric wards show legal status, risk review and psychiatric care tabs. Ward tabs are assigned by ward type.

### 8.4 Beds

`In-Patient > Beds Management` (`/inpatient/admissions/beds-management`) lists each bed with its ward, bed type and current occupants. **Release Bed** frees a bed. If every bed shows **Available**, there are no active admissions.

### 8.5 The charge sheet

The charge sheet lists every charge on an admission by day. From the discharge lists choose **Update Chargesheet**.

- Changing the price of a daily or ward charge applies across all days.
- Reducing the units of a daily or ward charge removes it from the last days.
- A charge already invoiced must be corrected on the invoice in Finance.
- You can add a **Rebate** to the charge sheet.
- **Finalize Chargesheet and submit**, then **Process the bill**.

### 8.6 Discharging a patient

1. In the chart's **Discharge** tab, the doctor writes the discharge summary and raises the discharge request.
2. `In-Patient > Discharge Requests` (`/inpatient/discharge-requests`) lists patients awaiting discharge with their account amount. Update and finalise the charge sheet and clear the bill.
3. Click **Discharge**. On the confirmation page check the account balance and discharge prescriptions, set the condition (Improved, Stable, Unstable, Deceased) and click **Confirm Patient Discharge**.
4. `In-Patient > All Discharges` has the printouts: **Discharge Summary**, **Gate Pass**, **Charge Sheet**, **Daily Charge Sheet**, **Discharge Prescriptions**, **Checkout Form**, and export of the charge sheet to Excel.

A discharge request can be withdrawn with **Cancel Discharge Request**. Permission to discharge: `inpatient.discharge_patients`.

`In-Patient > Admission Logs` is the full history of admissions, with statuses Admitted, Discharged, Transferred, Deceased, Absconded and Cancelled. `In-Patient > Pending Procedures` lists inpatient procedures awaiting approval.

### 8.7 Inpatient setup

Menu: `Setup > Inpatient`. Permission: `inpatient.manage_wards`.

| Screen | What you set |
|---|---|
| Wards | Ward number, **Ward Type**, display name, gender, age group, cash and insurance cost, whether it provides maternity care or a psychiatry service |
| Beds | **Bed Number**, **Ward**, **Bed Type** |
| Bed Types | The kinds of bed you offer |
| Admission Types | Name, **Deposit (Ksh)**, description, associated procedure |
| Discharge Types | The ways a patient can be discharged |
| Recurring & One-off Fees | Inpatient charges: name, cost, **One-Off Charge** or **Recurring Charge**, the wards they apply to, and **Post automatically on admission** |

---

## 9. Theatre

Menu: `Theatre`. Permission: `theatre.manage`.

### 9.1 Booking statuses

A theatre booking moves through: **Pending** → **Approved** (optional) → **Authorized** → **Booked** → **In Progress** → **Completed**. A booking can be **Rejected** (with a reason) or **Cancelled** at any point before it is In Progress.

| Status | Meaning | What you can do next |
|---|---|---|
| Pending | Just created | Approve, Authorize, set Payment Mode, Reject, Cancel |
| Approved | Approved by a manager | Authorize, Schedule, Cancel |
| Authorized | Payment authorized | Schedule, Cancel |
| Booked | Scheduled into a theatre slot | Ward checks, Send to theatre, Cancel |
| In Progress | Operation under way | Record the operation |
| Completed | Operation done | View only |

### 9.2 Step by step

1. **Create the booking.** `Theatre > Booking Requests`, then **New Booking** (`/theatre/bookings/create`). Choose the **Patient**, **Procedure**, **Preferred Surgeon**, **Theatre Deposit** and **Preferred Date**. Tick **Emergency Case** if needed. Choose **Cash** or **Insurance** (and the scheme). Click **Create Booking**.
2. **Authorize payment.** On the bookings list open **Actions**. **Authorize** asks for the **Authorization Amount (KES)**. You can authorize without approving first. **Payment Mode** records **Cash Deposit** or **Insurance Cover**. **Reject** needs a reason.
3. **Schedule.** Click **Schedule** (`/theatre/schedules/create`). Select the **operation room**, the **date**, **start time** and **end time**. "This timeslot is already booked" means you must pick another slot. Add **Requirements/Consumables** from a store if needed. The booking becomes **Booked**.
4. **Pre-operative checks.** On `Theatre > Patient Management` (`/theatre/schedules`) the **Ward/Theatre Check** column opens the **Patient Consent Form**, **Nurse Ward Check**, **Doctor Ward Check** and **Anaesthetist Checklist**. `Theatre > Theatre Checkin` (`/theatre/checkin`) lists booked patients and shows whether checks are done or pending.
5. **Send to theatre.** On the schedules list click **Send to theatre**. Enter the **Anaesthesia Type**, the **Pre-operative Diagnosis (ICD-11)** and the **Planned Procedure**. The booking becomes **In Progress**.
6. **Record the operation.** The operation screen has the operation form (start times, premedication, induction notes), drugs and prescriptions, the anaesthetics table (blood pressure, pulse, fluids), procedure notes, consumables and the nurse's **Sign Out**.
7. **Billing.** Theatre charges are collected at `Finance > Cash Workbench` using the **Theatre** toggle, or through the Insurance Workbench for insured patients.

### 9.3 Theatre setup

Menu: `Setup > Theatre Settings`. Permission: `theatre.setup`.

- **Theatre Types** — categories of theatre
- **Operation Theatres** — each room, with name, number, type, facility, associated procedure, charging cycle and fees
- **Theatre Deposits** — named deposit amounts per facility, selected when booking

---

## 10. Inventory and procurement

Menu: `Inventory & Procurement`. Permission: `inventory.manage`.

The inventory home page (`/inventory`) shows Total Products, Active Stores, Active Suppliers, Low Stock Items, today's stock in and out, and recent stock movements.

### 10.1 Product setup

Menu: `Inventory & Procurement > Products`.

Set these up first: **Tax Categories**, **Units of Measure**, **Product Categories**, **Brands**. Then add products in **Product Management** (`/inventory/products`) with **New Product**:

- **Generic Name**, product code, category, unit of measurement, brand and tax group
- **Cost**, **Selling Price** (the price of one sold unit, not of the big pack), **Insurance Price**, discount
- **Reorder Level** (drives low-stock alerts)
- **Dispensable Amount** (tablets or ml inside one sold unit), **Strength**, **Pack Size**
- Whether the product tracks batch, expiry or serial numbers

Prices can be changed in bulk with **Bulk Update Prices** or **Bulk Update Prices (CSV)**. **Print Product Labels** prints shelf labels.

If you switch on batch or expiry tracking for a product that already has stock, the system recommends selling or adjusting the existing stock to zero first.

### 10.2 Stores

Menu: `Inventory & Procurement > Stores`. `Create Store` needs permission `inventory.stock.create`.

A store has a **Store Name**, code, facility, department, optional **Parent Store** and a **Store Type**: Main, Pharmacy, General / Departmental Store, Ward, Theater, Laboratory, or Consumable. Options: **Main Store**, **Can order from suppliers**, **Can update product prices**, **Restrict opening hours** (with open and close time).

`List Stores`, then **Manage** on a store opens these tabs:

| Tab | Use |
|---|---|
| Products | Stock held in this store |
| Add Items | Add opening stock with quantity, unit cost, batch number, expiry date and serial numbers |
| Direct Orders | Receive goods straight from a supplier without a purchase order |
| Internal Orders | Request stock from another store; accept, dispatch or reject requests from other stores |
| Received Orders | Confirm receipt of stock sent from another store |
| Issue Products | Send stock to another store |
| Stock Take | Physical counts and reconciliation |
| Knock Off | Remove damaged, expired or defective stock |
| Analytics, Reports | Store-level figures |

### 10.3 The procurement cycle

Menu: `Inventory & Procurement > Procurement`.

1. **Requisition.** **New Requisition** (`/inventory/procurement/requisitions/create`). Choose the type — **Purchase (From Supplier)** or **Internal (Store Requisition)** — the requesting store, the supplying store or preferred supplier, the required-by date and priority. Add products and quantities. **Save as Draft** or **Submit for Approval**.
2. **Approval.** An approver opens the requisition and uses **Approve** (setting the quantity to approve) or **Reject** (a reason is required). Several can be handled with **Bulk Approve** / **Bulk Reject**. Approvers and their order are set in `Setup > Inventory Settings > Approval Levels`.
3. **Purchase order.** On an approved purchase requisition click **Convert to PO**, or create one directly with **New Purchase Order**. Choose the **Supplier**, **Receiving Store**, dates, payment terms, then add items with quantity, unit cost and discount. **Submit for Approval**; once approved use **Send to Supplier** or **Email LPO to Supplier**.
4. **Goods received.** When goods arrive click **Receive Goods** on the purchase order. A **Goods Received Note (GRN)** records what was ordered, received, accepted and rejected, with batch number, expiry date and serial numbers.
5. **Approve the GRN.** **Mark as Inspected**, then **Approve & Update Stock**. Stock in the store increases only when the GRN is approved.

Purchase order statuses: Draft, Pending Approval, Approved, Partially Approved, Sent, Accepted by Supplier, Partially Received, Received, Rejected, Cancelled.

### 10.4 Moving stock between stores

- **Requesting store:** `Manage > Internal Orders`, choose **Request from Store**, add products and click **Make Order**.
- **Supplying store:** under **Incoming Requests**, **Accept** then **Dispatch** (confirming quantities, batch and expiry), or **Reject**.
- **Requesting store again:** `Manage > Received Orders`, click **Receive** and **Confirm Receipt**.

A store can also push stock without a request from `Manage > Issue Products`.

### 10.5 Stock take

`Manage > Stock Take`, then **New Stock Take**. Add products (search, **Add All in Store** or **Add All in Category**), then enter the **Physical Count** against the **System Qty** for each batch, with a note for any variance. Counts can be typed in or loaded with **CSV Upload**. **Submit** the stock take to reconcile stock.

### 10.6 Knocking off stock

`Manage > Knock Off`. Search for the product, enter the quantity and the **Reason** (for example expired or damaged) and confirm. This permanently reduces stock, so check quantities first. Serial numbers are required for products that track them.

### 10.7 Suppliers

`Inventory & Procurement > Suppliers`. **Add New Supplier** creates one. **View Account** on a supplier shows a summary, purchase orders, direct orders, payments and a statement.

### 10.8 Inventory reports

`/inventory/reports`: Stock Expiry Report, Stock Movement Report, Purchase Order Report, GRN Report, Low Stock Report, Store Valuation Report.

---

## 11. Point of Sale (POS)

Menu: `Point of Sale`. Permission: `pos.manage`; selling needs `pos.sales.create`.

POS is for direct sales, such as a pharmacy walk-in sale that is not part of a clinical visit.

### 11.1 Setting up a point of sale

`Setup > Pos setup` (`/settings/pos/management`). Permission: `pos.terminals.manage`. For each point of sale set the **Display Name**, **Patient type**, **Pos type**, the **store** it sells from, the **facility**, and options such as **Is it a pharmaceuticals pos**, **Has stocks involvement**, **Take payment at this till** and **Active**.

Each user must also be authorised for the stations they use: open the user, then **Authorized POS Stations**. The stations a user is authorised for appear under Point of Sale in their sidebar.

### 11.2 Making a sale

1. Open your station from the sidebar.
2. **Open a shift:** enter the **Opening Cash Amount** and click **Open Shift & Enter**.
3. Choose the customer: **Walk-in Customer**, **Existing patient** (search by name, mobile or ID number) or **Staff member**.
4. Choose **Cash / Mpesa / Card** or **Insurance** (and the scheme).
5. Select the store, search for each product and add it to the **Cart**. For medicines enter the dosage, frequency and duration.
6. Complete the sale and take payment, or send it to the Cash Workbench to be paid there, depending on how the station is set up.

### 11.3 Closing a shift

Click **Close Shift**, count the drawer and enter the **Closing Cash Count**. The system shows **Expected in Drawer**, **Counted** and the **Variance**. `Point of Sale > Shifts` (`/pos/shifts`) lists all shifts with opening and closing cash and variance, and prints a shift receipt.

### 11.4 Sales history, voids and reports

- `/pos/sales` — search receipts, **View Sale**, **Print Receipt**, **Export to Excel**. **Void Sale** needs a reason; a voided sale returns its stock.
- `/pos/reports` — total revenue, transaction count, average sale, voided amount, totals by payment method and a daily summary, for Today, Yesterday, This Week, This Month or a custom range.

---

## 12. Maternal and child health (MCH)

Menu: `MCH`. Permission: `evaluation.manage`.

Each clinic has three screens: **Enrollment**, **Evaluation** and **Register**.

| Clinic | Enrollment is based on | Register |
|---|---|---|
| ANC (antenatal) | LMP and EDD | ANC Register (MOH 405) |
| PNC (postnatal) | Delivery date | PNC Register (MOH 405) |
| CWC (child welfare) | Date of birth | CWC Register (MOH 510) |

**How enrollment works**
1. A patient is put in the awaiting-enrollment queue from the vitals screen (**Enroll to ANC**, **Enroll to CWC Clinic**, or the postnatal option).
2. Open the clinic's **Enrollment** screen. The **Awaiting Enrollment** tab lists those patients. Click **Enroll**.
3. The system proposes a visit schedule. Review it and click **Enroll**.
4. The **Enrolled** tab lists enrolled patients. **View / Edit Schedule** changes the dates; **Check In** starts a visit for a scheduled contact; **Discontinue** ends enrollment with a reason.

**Family planning** (`/mch/fp`): **Record FP Service** adds a client to the Family Planning Register, and the page produces the MOH 711 Section D summary.

---

## 13. Renal / Dialysis

Menu: `Renal/Dialysis`. Permissions: `dialysis.sessions.view`, `dialysis.patients.view`, `dialysis.schedules.view`.

### 13.1 Setup

`Setup > Dialysis Settings` (permission `dialysis.setup`): Dialysis Plants, Ward Types, Wards, Machines, Chairs, Vascular Access Types, Coagulation Types and Dialysis Deposits. At least one machine and chair must exist before sessions can be scheduled.

### 13.2 Patient pathway

1. **Enroll the patient.** `Renal/Dialysis > Patients`. Search for the registered patient and record sessions per week, hours per session, plant, ward, assigned machine and chair, access type, dry weight, blood type, start date and diagnosis.
2. **Give the patient a weekly slot.** `Schedules > Weekly Slots`, **Add Weekly Slot**: patient, chair, day of week, start and end time. **Generate Bookings** creates the bookings from the weekly slots.
3. **Or book a single appointment.** `Schedules`, **Schedule Appointment**.
4. **On the day.** `Overview` shows **Today's Sessions**. **Check In** the patient, then **Start** the session.
5. **During the session.** The session screen records vascular access, pre-dialysis assessment, and regular observations: blood pressure, pulse, temperature, blood flow rate, UF rate, venous pressure, TMP, dialysate flow and temperature, and heparin dose.
6. **Complete.** Record post-dialysis weight, blood pressure and pulse, UF achieved and notes, then **Discharge**. Print the **Discharge summary**.

Session statuses: Scheduled, In Progress, Completed, Discharged, Cancelled, Missed. `Renal/Dialysis > Calendar` shows the schedule as a calendar, and `Discharged Sessions` lists finished sessions.

---

## 14. Oncology

Menu: `Oncology`. Four areas, each with its own permission.

| Area | Screens | Use |
|---|---|---|
| Screening | Awaiting enrollment, Screening register, Due list, Referrals out | Record cancer screenings, see who is due, track referrals |
| Oncology Clinic | Caseload, then the patient record | Diagnosis, staging, tumour board, treatment plan, toxicity, SHA benefit |
| Chemo Day Unit | Chairs today, then a treatment cycle | Pre-administration checks, doses and the administration record |
| Palliative Care | Care and follow-up | Palliative caseload, surveillance and tracing of missed visits |

**Screening.** **New screening** records the programme, the method, the date and the result (Negative, Positive, Unsatisfactory) and exactly one action: treat here, schedule a session, or refer out. The next due date is calculated from the programme's interval. The **Due list** shows who is overdue and can send reminders to patients who consented to messages. **Referrals out** tracks each referral until a result is returned.

**Oncology clinic.** The patient record has: Diagnosis (site, morphology, staging), Tumour board (the decision and recommended sequence), Treatment plan (protocol, intent, cycle schedule, computed doses, projected cost), Toxicity and response, and SHA benefit and claims. A diagnosis is required before a treatment plan can be built, and a plan must be approved before cycles open.

**Chemo day unit.** A cycle shows its **Pre-administration gates** — the checks the protocol requires. If a check fails, opening the cycle and giving the drug are blocked, and the screen lists what must be resolved. Recording observations, vitals, toxicity and notes is never blocked. The administration record needs two different people: the person who prepared the drug and the person who verified it.

The system calculates eligibility, schedules, doses and benefit figures. Staff do not enter these by hand.

---

## 15. Psychiatry and rehabilitation

Menu: `Psychiatry & Rehab`. Permissions begin with `psychiatry.`; the general clinical permission does not give access.

- There are three services: **Mental Health Services**, **Rehabilitation Services** and **Therapeutic Services**. Each has a **Caseload** and **Follow-ups**. A user sees only the services they are authorised for.
- Psychiatric records are restricted. Opening one asks you to **Confirm your password** before the record unlocks, and every access is logged. The record shows who has read it.
- A psychiatric record is started from a visit: reception checks the patient in to Psychiatry, then the clinician clicks **Start intake**.
- The record covers episodes, assessments (history, mental state examination, risk reviews, scales), legal status and admission forms, the care plan, therapy sessions and aftercare.
- **Alerts** can be acknowledged, escalated (a named person is required) or resolved.
- **Emergency access review** (`Setup > Psychiatry Settings`) is where emergency ("break-glass") access to a restricted record is reviewed and given a verdict: Appropriate, Needs follow-up or Inappropriate.
- `Setup > Psychiatry Settings > Psychiatry` controls which forms the clinic uses, the wording shown to patients in reminders and printouts, the crisis line, and safety rules such as blocking discharge until the discharge summary is signed.

---

## 16. Morgue

Menu: `Morgue > Mortuary`.

**Setup first.** `Morgue > Settings`: **Add a fridge** with a name, code, kind (Fridge, Freezer or Transit), target temperature, location and the number of spaces. A fridge must exist before a body can be placed.

| Screen | Use |
|---|---|
| Expected arrivals | Deaths notified from the wards. **Receive** when the body arrives, or **Never came** with what happened |
| Receive a body | Record the source, the deceased's identity, date, time, place, type and provisional cause of death, police report details, who brought the body, condition on arrival and personal effects, and choose a storage space |
| Occupancy | Every fridge and space; take a space out of service or return it |
| Register | All bodies, by status: In storage, Released, Transferred out, Disposed |
| Releases | Raise, approve and complete the release of a body |
| Unclaimed | Bodies held a long time, with the steps taken to trace next of kin |
| Billing | Mortuary charges and what is outstanding; hand charges to finance |

**Releasing a body.** Open the record and click **Raise a release**, recording the release type, permit, collector and collector's ID. A different user must **Approve** it — the person who raised a release cannot approve it. Then **Hand over** completes it and a **Handover sheet** can be printed.

A body can be received without a space and placed later. Each personal effect is tracked separately until released. Mortuary charges are paid at `Finance > Cash Workbench` using the **Morgue** toggle.

---

## 17. SHA (Social Health Authority)

Menu: `SHA`. The section appears only when SHA integration is switched on for the organization.

| Screen | Page address | Use |
|---|---|---|
| Claims | `/evaluation/sha/claims` | Open SHA visits and their claim status |
| Submitted claims | `/evaluation/sha/claims/submitted` | Claims with SHA: approved, rejected, sent back, paid |
| Preauthorisations | `/evaluation/sha/preauths` | Preauthorisation requests and decisions |
| Shared Health Record (SHR) | `/evaluation/sha/shr` | Visits shared to the national health record |
| SHA Callbacks | `/evaluation/sha/callbacks` | Messages received back from SHA |

**Before a claim can be submitted**, the claim page shows a checklist. Each of these must be satisfied:
- At least one protocol billed, with a diagnosis on it
- At least one diagnosis added, with laterality on every sided diagnosis
- Doctor details on the claim
- At least one bill line added
- The claim previewed at least once
- An invoice number selected
- Preauthorisation in order, where the intervention needs one
- A valid discharge date (not in the future)

**Time limit.** An unsubmitted claim becomes **Time barred** when it passes 7 days. The claims list shows the days remaining and has a filter for unsubmitted claims that can still be sent.

**Requirements elsewhere in the system for SHA to work:** the facility has its SHA FR Code (3.2); the patient's SHA scheme and member details are on their record (5.1); the visit was checked in under the SHA scheme with the SHA services chosen (5.5); and the doctor recorded an ICD-11 diagnosis (6.5).

Preauthorisations are raised from a visit's claim page. Their decisions are Approved, Rejected, Clarification needed or Pending with SHA.

---

## 18. Laboratory setup

Menu: `Setup > Lab Settings`. Permission: `lab.setup`.

Set the laboratory up in this order. Steps 1 to 6 are lookup lists that the tests in step 7 refer to.

### 18.1 Setup order

| Step | Screen | Page address |
|---|---|---|
| 1 | Laboratory Location (check the lab destination exists) | `/evaluation/setup/lab-destination` |
| 2 | Test Categories | `/evaluation/setup/lab-test-categories` |
| 3 | Test Units | `/evaluation/setup/test-units` |
| 4 | Sample Types, Collection Methods, Additives/Containers | `/evaluation/setup/sample-types`, `/evaluation/setup/collection-methods`, `/evaluation/setup/lab-test-additives` |
| 5 | Age Groups | `/evaluation/setup/age-groups` |
| 6 | The lab tests themselves, as procedures | `/evaluation/setup/procedures/create` (section 3.5) |
| 7 | Test Titles and result templates for panels | `/evaluation/setup/test/titles` |
| 8 | Biological References (normal ranges) | `/evaluation/setup/reference-ranges` |
| 9 | Critical Values | `/evaluation/setup/critical-values` |
| 10 | Remark Templates | `/evaluation/setup/test-remarks` |
| 11 | Formulae for calculated results | `/evaluation/setup/formulae` |
| 12 | Machines and code mapping, if analysers are connected | `/evaluation/setup/machine` |
| 13 | Result Analytics Groups | `/evaluation/setup/result-analytics-groups` |

### 18.2 Laboratory Location

This screen shows which destination is the laboratory for the facility. It is read-only. If it says "No destination is configured as the Laboratory for this facility yet", create a destination and tick **Is a lab queue** (section 3.3).

### 18.3 Lookup lists

| Screen | Fields |
|---|---|
| Test Categories | **Name** *, Description. For example Haematology, Biochemistry, Microbiology |
| Test Units | **Name** *, Abbreviation (for example mg/dL), Description |
| Sample Types | **Name** *, Tube Color (for example Red, Purple, Grey), Volume Required, Storage Temp, Details, Active |
| Collection Methods | **Name** *, Description, Active |
| Additives/Containers | **Additive Name** *, Additive Description |
| Age Groups | Lower value and Upper value, with the unit: Years, Months, Weeks or Days. Used by reference ranges and critical values |

### 18.4 Lab tests and panels

Each lab test is a procedure (section 3.5). For a panel such as a Full Haemogram:

1. Create the panel as a procedure.
2. Create each component as its own procedure with the panel as its **Parent procedure**, and give each a **Result Type**, **Unit of measure** and **Sample Type**.
3. On the panel choose **Template** to arrange the sub-tests in the order they should appear on the result, with optional titles and bold lines. Templates apply to tests with sub-tests; a test with no sub-tests uses the input for its own result type.
4. Use **Test Titles** to add sub-headings inside a result template (name, the procedure, sort order and sub text such as "Serum").

### 18.5 Biological References (normal ranges)

For each test, and where needed for each gender and age group, define the normal range:

- **Parent Procedure** and **Sub Procedure**
- **Gender** (Male, Female or Both) and **Age Group**
- **Type:** Range (with Lower Value and Upper Value), Less / Greater Than (with a comparator: less than, greater than, equal to, less than or equal to, greater than or equal to), or Text
- **Flag** — the label shown, for example "Desirable"
- **Mark as normal range**

Many ranges can be loaded at once with **Bulk Import (CSV)**: download the template, fill it in and import it. The template uses the procedure names and ids shown on screen, for example "Full Haemogram(41)".

Results outside the normal range are flagged on screen and on the printed report, provided the test has **Use biological reference flagging** ticked.

### 18.6 Critical Values

A critical value is a result that needs urgent attention. For each: the **Procedure** *, **Critical When** * (the comparison), the **Critical Value** *, and optionally gender and age group. Bulk import by CSV is available, with a template.

### 18.7 Remark Templates

Standard remarks that can be attached to a result. For each: the **Procedure** *, optional gender and age group, a **Title** to show on the result, the remark text, and **Editable on result posting** if staff may change it when entering results.

### 18.8 Formulae

For results calculated from other results. Choose the **Test** and enter the **Formula**. Formulae follow spreadsheet format and refer to tests by their procedure id with a P in front: `P100*P13` means the result of procedure 100 multiplied by the result of procedure 13. Search for procedure ids on the screen; they are shown in brackets.

### 18.9 Machines (analyser integration)

1. Click **Add Machine**. Enter the **Display Name** *, the **Machine Identifier**, and the provider's name, mobile and email. The Machine Identifier must be exactly the name the machine itself sends.
2. On the machine choose **Map Codes**. For each test the machine reports, enter the **Machine test name** and **Machine test code** and choose the matching **Procedure**.

Results received from a mapped machine appear under **Machine Results** on the patient's lab screen. The latest reading is used by default and staff can choose another.

### 18.10 Laboratory behaviour controlled by settings

| Behaviour | Setting key |
|---|---|
| Lab and radiology tests wait for payment before they can be worked on | `evaluation.enforce_payment_before_lab_radiology_testing` (on by default) |
| Let patients receive investigations without going through billing | `evaluation.bypass_investigations_billing` (editable in General Settings) |
| Offer inpatient investigations before payment | `evaluation.offer_inpatient_investigations_before_payment` (on by default) |

---

## 19. Reports

Menu: `Reports`. Permission: `reports.view`. The reports landing page (`/reports`) has a search box (try "cashier", "MOH 705", "stock", "dialysis").

Reports are grouped as: Reception Reports, Outpatient Reports, Inpatient Reports, Financial Reports, Inventory Reports, MOH Reports (including MCH Clinic), Dialysis Reports, SHA Reports, Users Reports and Other Reports. The list comes from the server, so new reports appear automatically.

| Group | Reports |
|---|---|
| Reception | Patients List, Patients Seen, Departmental Visits, Patient Contacts |
| Outpatient / clinical | Outpatient Evaluation, Facility Summary, Lab Turnaround Time, Patient Results Analytics |
| MOH registers | MOH 405, MOH 510, MOH 511 |
| Finance | Cash Receipts, Cash Summary, Cashier Collections, Cashier Summary, Credit Notes, Department Summary, Drugs Dispensed, Exemptions, Inpatient Revenue, Ward Revenue, Insurance Status, Insurance Summary, Invoices, Lab Performance, Radiology Performance, Lost Sales, Revenue Summary, Shop Report, Waivers, Walk-in Reports |
| Inventory | Stock Movement, plus the reports in section 10.8 |
| SHA | Claims Workbench, HIE Transactions, Preauthorisations |
| Users | Staff Allocation |
| Dialysis | Dialysis reports |

Most reports have date filters and an export to Excel or a print option.

---

## 20. Settings reference

Settings are the organization-wide switches and values that change how the system behaves. They are viewed in `Setup > General Settings` (section 3.8).

How to use this reference:
- The **key** is the setting's internal name. Quote it when asking support to change a setting.
- **Default** is the value the application uses when the organization has never saved the setting. "—" means the application has no built-in default and treats an unsaved setting as off or empty.
- The **effect** is described from the setting's name and where the application uses it. Confirm the exact behaviour with support before changing a setting that affects billing or clinical work.
- Most settings are locked on screen and are changed by Afya Analytics support. The ones an administrator can change are marked **Editable**.

### 20.1 Core and sidebar

| Key | Default | Effect |
|---|---|---|
| `core_sidemenu_pos` | On | Show the Point of Sale sidebar section |
| `core_sidemenu_sms` | On | Show the SMS section |
| `core_sidemenu_reception` | On | Show Reception & Registration |
| `core_sidemenu_outpatient` | On | Show OutPatient |
| `core_sidemenu_inpatient` | On | Show In-Patient |
| `core_sidemenu_mch` | On | Show MCH |
| `core_sidemenu_theatre` | On | Show Theatre |
| `core_sidemenu_oncology` | On | Show Oncology |
| `core_sidemenu_psychiatry` | On | Show Psychiatry & Rehab |
| `core_sidemenu_morgue` | **Off** | Show Morgue |
| `core_sidemenu_dialysis` | On | Show Renal/Dialysis |
| `core_sidemenu_inventory` | On | Show Inventory & Procurement |
| `core_sidemenu_finance` | On | Show Finance |
| `core_sidemenu_sha` | On | Show SHA (also needs `sha_integration`) |
| `core_sidemenu_reports` | On | Show Reports |
| `core_site-display-name` | empty | The display name of the site |
| `core_disable_mandatory_triage` | On | Triage is not forced before a patient can be seen |
| `core_disable_password_check` | Off | Skip the password confirmation on protected actions |
| `core_enable_commercial_settings_core` | — | Enable commercial (non-clinical) options |
| `core_pos.default_walkin_customer` | On | A POS sale defaults to a walk-in customer |

### 20.2 Reception

| Key | Default | Effect |
|---|---|---|
| `reception.skip_registration_consent` | — | Skip the consent step when registering a patient |
| `reception.remove_middle_name` | — | Hide Middle Name on the patient form |
| `reception.remove_town` | — | Hide Town |
| `reception.remove_home_location` | — | Hide Home Location |
| `reception.remove_occupation` | — | Hide Occupation |
| `reception.remove_county_of_residence` | — | Hide County of residence |
| `reception.remove_sub_county` | — | Hide Sub county of residence |
| `reception.remove_village` | — | Hide Resident Village/Estate |
| `reception.number_of_dependants` | No limit | Maximum dependants per patient |
| `reception.make_clinicians_checking_mandatory` | — | A clinician must be chosen at check-in |
| `reception.performing_doctor_compulsory` | — | A performing doctor must be chosen for pre-paid procedures |
| `reception.enforce_prepaid_fees` | — | Pre-paid fees must be chosen at check-in |
| `reception.appointments_per_hour` | — | Limit on appointments per hour |
| `reception.checkin_to_roles` | empty | Restricts which staff roles patients can be checked in to |

Field visibility and mandatory rules for the patient form are set in `General Settings > Registration` (section 3.8).

### 20.3 Evaluation (outpatient clinical work)

| Key | Default | Effect |
|---|---|---|
| `evaluation.enforce_payment_before_lab_radiology_testing` | On | Lab and radiology tests wait for payment |
| `evaluation.bypass_investigations_billing` | Off | **Editable.** Investigations can be done without going through billing |
| `bypass_prescriptions_billing` | — | **Editable.** Drugs can be given without going through billing |
| `evaluation.bypass_procedures_billing` | — | Procedures can be done without going through billing |
| `evaluation.offer_inpatient_investigations_before_payment` | On | Inpatient investigations are done before payment |
| `evaluation.enable_ordering_procedures_without_price` | — | A procedure with no price can still be ordered |
| `evaluation.auto_checkout` | — | Patients are checked out automatically |
| `evaluation.presenting_complaints_compulsory` | — | Presenting complaints must be recorded |
| `evaluation.presenting_complaints_before_selection` | — | Presenting complaints are captured before other selections |
| `evaluation.write_in_diagnosis` | — | The doctor may type a diagnosis as free text |
| `evaluation.allow_doctors_to_type_prescriptions` | — | The doctor may type a drug that is not in the product list |
| `evaluation.enable_adding_of_batch_no_when_doc_requesting` | — | A batch number can be chosen when the doctor requests a drug |
| `evaluation.show_all_patient_notes` | Off | Show notes from all clinicians rather than only the current one |
| `evaluation.vitals_panel_in_doc_notes_open_by_default_instead_of_prescription_panel` | — | In doctor's notes the vitals panel opens first |
| `evaluation.use_centimetres_instead_of_metres_under_vitals` | — | Height is entered in centimetres |
| `evaluation.enable_paid_sick_chit` | On | The Paid Sick Chit is available under Documents |
| `evaluation.enable_light_duty_chit` | On | The Light Duty Chit is available |
| `evaluation.enable_referral_chit` | On | The Referral Chit is available |
| `evaluation.enable_treatment_work_back_chit` | On | The Treatment and Back to Work Chit is available |
| `evaluation.enable_jspaint` | — | The drawing tool is available in notes |
| `evaluation.optical_pos` | — | The optical point of sale is used for eyewear |
| `evaluation.with_clinic_renal` | — | The renal clinic options are shown |
| `evaluation.remove_theatre_inpatient_renal_diagnosis_bookings` | — | Hide the theatre, inpatient and renal booking buttons in the patient header |

### 20.4 Finance

| Key | Default | Effect |
|---|---|---|
| `finance.enable_cash` | On | Cash is offered as a payment mode |
| `finance.enable_card` | On | Card is offered |
| `finance.mpesa_integrated` | On | Integrated M-Pesa is offered |
| `finance.mpesa_non_integrated` | On | Manual M-Pesa (reference typed in) is offered |
| `finance.pesa_pal_card_integrated` | On | Pesapal card is offered |
| `finance.pesa_pal_mpesa_integrated` | On | Pesapal M-Pesa is offered |
| `finance.wallet_integrated` | — | Wallet payment is offered |
| `finance.enable_patient_account` | On | Patient accounts (deposits and balances) are used |
| `finance.enable_discount_payment_mode` | On | Discount is available when taking payment |
| `finance.enable_discounts_for_cash_payers` | Off | Cash payers can be given discounts |
| `finance.max_discount_percentage` | 100 | The largest discount allowed, as a percentage |
| `finance.disable_cancel_invoice_button` | — | Hide the Cancel button on invoices |
| `finance.service_charges_default_one_off` | On | New service charges default to one-off |

### 20.5 Inpatient

| Key | Default | Effect |
|---|---|---|
| `inpatient.auto_generate_inpatient_no` | On | IP numbers are generated automatically |
| `inpatient.inpatient_no_prefix` | — | The prefix used on IP numbers |
| `inpatient.allow_overbooking_of_wards` | — | More than one patient can be put on a bed |
| `inpatient.allow_edit_of_discharged_patients_chargesheets` | — | A discharged patient's charge sheet can still be edited |

### 20.6 Inventory and POS

| Key | Default | Effect |
|---|---|---|
| `inventory.currency` | KES | The currency code |
| `inventory.currency_symbol` | Ksh | The currency symbol shown on screens and printouts |
| `inventory.enable_dispensable_amount` | On | Products have a dispensable amount (tablets per unit) |
| `inventory.can_approve_own_order` | — | A user may approve an order they raised |
| `inventory.show_dispatched_orders` | — | Dispatched orders are listed |
| `inventory_enable_discount_pricing` | — | Discount pricing is available on products |
| `inventory.pos_allow_zero_opening_float` | On | A POS shift can be opened with zero cash |
| `no_pos.receipt_paper_size` | 80mm | Receipt paper size |
| `no_pos.receipt_header_text` | empty | Text printed at the top of receipts |
| `no_pos.receipt_footer_text` | "Thank you for your purchase!" | Text printed at the bottom of receipts |
| `no_pos.receipt_show_logo` | Off | Print the logo on receipts |
| `no_pos.receipt_show_station` | On | Print the station name on receipts |

### 20.7 Theatre

| Key | Default | Effect |
|---|---|---|
| `theatre_payment_method` | inclusive | How theatre is charged: inclusive, one-off or per-cycle |
| `theatre.require_nurse_ward_check` | On | The nurse ward check is required before theatre |
| `theatre.require_doctor_ward_check` | On | The doctor ward check is required before theatre |
| `theatre.disable_wardcheck_printout` | — | Hide the ward check printout |

### 20.8 SHA

| Key | Effect |
|---|---|
| `sha_integration` | **Editable.** Master switch for SHA |
| `sha_default_scheme_id` | **Editable.** The scheme that represents SHA |
| `sha_default_intervention_code` | **Editable.** The default intervention code |
| `sha_agent_id` | **Editable.** National ID of the facility's HealthID biometrics agent |
| `sha_preauth_notification_email` | **Editable.** The inbox for preauthorisation updates |
| `sha_test_mode`, `sha_sandbox_mode`, `sha_sandbox_otp`, `sha_sandbox_practitioner_id`, `sha_sandbox_facility_code` | Test and sandbox values. Shown blurred, never editable on screen, and ignored in production |

### 20.9 Users

| Key | Default | Effect |
|---|---|---|
| `users.enable_check_roll_number` | Off | Show the Check-Roll Number field on staff records |

Psychiatry has its own settings screen (section 3.18), and registration form fields are covered in section 3.8.

---

## 21. Permissions reference

A permission is a code such as `reception.patients.view`. Permissions are grouped into roles, and roles are given to users (section 3.4). A user can also be given single permissions under **Special Permissions**. Users with the `superadmin` or `admin` role have wider access.

### 21.1 Permission needed for each sidebar entry

| Sidebar entry | Permission |
|---|---|
| Dashboard | `dashboard.view` |
| Point of Sale section | `pos.manage`; each station needs `pos.sales.create` |
| SMS | `sms.manage` |
| Reception section | `reception.manage` |
| Reception > Manage Patients, Calendar, Patients Follow up, Patient Docs | `reception.patients.view` |
| Reception > Manage Queue | `reception.visits.view` |
| Reception > Appointments | `reception.appointments.view` |
| OutPatient section | any of `evaluation.manage`, `lab.tests.view`, `pharmacy.dispensing.view`, `triage.view`, `triage.perform` |
| In-Patient section, Checkins/IPD, Patient Management, Pending Procedures, All Discharges, Admission Logs | `inpatient.admissions.view` |
| In-Patient > Admission Requests, Awaiting Authorization | `inpatient.admit_patients` |
| In-Patient > Beds Management | `inpatient.beds.view` |
| In-Patient > Discharge Requests | `inpatient.discharge_patients` |
| MCH (all entries) | `evaluation.manage` |
| Theatre section | `theatre.manage` |
| Theatre > Theatre Checkin | `theatre.checkin` |
| Theatre > Booking Requests | `theatre.bookings.index` |
| Theatre > Patient Management | `theatre.schedule.index` |
| Oncology > Screening | `oncology.screening.record` |
| Oncology > Chemo Day Unit | `oncology.cycle.open` or `oncology.cytotoxic.handle` |
| Oncology > Palliative Care | `oncology.toxicity.write` or `oncology.screening.record` |
| Renal/Dialysis > Overview, Sessions, Discharged Sessions | `dialysis.sessions.view` |
| Renal/Dialysis > Patients | `dialysis.patients.view` |
| Renal/Dialysis > Schedules, Weekly Slots, Calendar | `dialysis.schedules.view` |
| Inventory section | `inventory.manage` |
| Inventory > Products | `inventory.items.view`; Tax and Product Categories need `inventory.categories.view` |
| Inventory > Procurement > Requisitions | `inventory.requisitions.view` |
| Inventory > Procurement > Purchase Orders | `inventory.purchase_orders.view` |
| Inventory > Procurement > Goods Received Notes, Direct Order GRNs | `inventory.grn.view` |
| Inventory > Stores | `inventory.stock.view`; Create Store needs `inventory.stock.create` |
| Inventory > Suppliers | `inventory.suppliers.view`; Add New Supplier needs `inventory.suppliers.create` |
| Psychiatry section | any of `psychiatry.access`, `psychiatry.risk.view`, `psychiatry.reports.view` |
| Psychiatry > Mental Health / Rehabilitation / Therapeutic Services | `psychiatry.mental_health.access` / `psychiatry.rehabilitation.access` / `psychiatry.therapeutic.access` |
| Psychiatry > Caseload | `psychiatry.record.view` |
| Psychiatry > Follow-ups | `psychiatry.risk.view` |
| Finance section | `finance.manage` |
| Finance > Cash Workbench | `finance.process_payments` |
| Finance > Insurance Workbench, Credit Notes, Patient Invoices | `finance.invoices.view` |
| Finance > Insurance Management | `insurance.view` |
| Finance > Cash Payments | `finance.payments.view` |
| Finance > Sales Summary | `reports.view_financial_reports` |
| Finance > Patient Accounts | `finance.accounts.view` |
| SHA > Claims, Preauthorisations | any of `insurance_claims.submit`, `finance.manage`, `evaluation.manage` |
| SHA > Submitted claims | any of `insurance_claims.view`, `insurance_claims.submit`, `insurance_claims.resubmit`, `finance.manage`, `evaluation.manage` |
| SHA > Shared Health Record | `evaluation.manage` |
| Users & Staff section | `users.manage` |
| Users > Create New | `users.users.create` |
| Users > View Existing, Categories, Specialities | `users.users.view` |
| Users > User Roles | `users.roles.view` |
| Setup section | `settings.manage` |
| Setup > Theatre Settings | `theatre.setup` |
| Setup > UI Colors, Procedures Settings | `settings.edit` |
| Setup > Pos setup | `pos.terminals.manage` |
| Setup > General Settings, Appointment Categories | `settings.view` |
| Setup > HIS Client Settings | `settings.manage_locations` |
| Setup > Insurance/ Corporate | `insurance.view` |
| Setup > Finance (Banks, Service Charges) | `finance.view_financials` |
| Setup > Dialysis Settings | `dialysis.setup` |
| Setup > Psychiatry Settings | `psychiatry.settings.manage`; Emergency access review needs `psychiatry.break_glass.review` or `psychiatry.access_log.view` |
| Setup > Inpatient | `inpatient.manage_wards` |
| Setup > Lab Settings | `lab.setup` |
| Setup > Sudo > Audit Trail | `audit.view` |
| Reports | `reports.view` |

### 21.2 Permission codes by module

| Module | Permission codes |
|---|---|
| Dashboard | `dashboard.view` |
| Reception | `reception.manage`, `reception.patients.view`, `reception.patients.create`, `reception.visits.view`, `reception.queue.manage`, `reception.appointments.view`, `reception.appointments.delete` |
| Triage and clinical | `triage.view`, `triage.perform`, `evaluation.manage`, `clinical_notes.sign`, `clinical_history.record`, `procedures.request`, `procedures.view` |
| Laboratory | `lab.tests.view`, `lab.samples.view`, `lab.samples.collect`, `lab.samples.receive`, `lab.samples.process`, `lab.results.view`, `lab.results.verify`, `lab.setup`, `investigations.record_results`, `investigations.verify_results` |
| Pharmacy | `pharmacy.dispensing.view`, `pharmacy.dispensing.dispense`, `pharmacy.prescriptions.view`, `pharmacy.prescriptions.fill`, `prescriptions.view`, `prescriptions.update`, `prescriptions.dispense` |
| Inpatient | `inpatient.manage`, `inpatient.admissions.view`, `inpatient.admit_patients`, `inpatient.authorize_without_payment`, `inpatient.discharge_patients`, `inpatient.beds.view`, `inpatient.wards.view`, `inpatient.manage_wards`, `inpatient.psychiatry_gate.override`, `wards.view`, `wards.create`, `wards.update`, `wards.delete`, `beds.create`, `beds.update`, `beds.delete` |
| Theatre | `theatre.manage`, `theatre.checkin`, `theatre.bookings.index`, `theatre.bookings.view`, `theatre.bookings.create`, `theatre.bookings.approve`, `theatre.bookings.reject`, `theatre.bookings.cancel`, `theatre.schedule.index`, `theatre.schedule.view`, `theatre.schedule.store`, `theatre.wardcheck.view`, `theatre.wardcheck.nurse`, `theatre.wardcheck.doctor`, `theatre.wardcheck.anaesthetist`, `theatre.operations.view`, `theatre.billing.view`, `theatre.payments.process`, `theatre.deposits.index`, `theatre.deposits.manage`, `theatre.theatre.index`, `theatre.types.index`, `theatre.setup` |
| Finance | `finance.manage`, `finance.process_payments`, `finance.payments.view`, `finance.invoices.view`, `finance.accounts.view`, `finance.view_financials`, `invoices.view`, `invoices.void`, `invoices.delete`, `credit_notes.apply`, `credit_notes.approve`, `credit_notes.void`, `waivers.approve`, `deposits.apply`, `billing.view`, `billing.manage`, `billing.configure_rates` |
| Insurance and SHA | `insurance.view`, `insurance_claims.view`, `insurance_claims.submit`, `insurance_claims.resubmit` |
| Inventory | `inventory.manage`, `inventory.items.view`, `inventory.categories.view`, `inventory.stock.view`, `inventory.stock.create`, `inventory.manage_stock`, `inventory.adjust_stock`, `inventory.transfer_stock`, `inventory.manage_stores`, `inventory.stores.create`, `inventory.stores.update`, `inventory.stores.delete`, `inventory.suppliers.view`, `inventory.suppliers.create`, `inventory.suppliers.update`, `inventory.requisitions.view`, `inventory.requisitions.create`, `inventory.requisitions.update`, `inventory.approve_requisitions`, `inventory.purchase_orders.view`, `inventory.purchase_orders.create`, `inventory.purchase_orders.approve`, `inventory.purchase_orders.cancel`, `inventory.purchase_orders.delete`, `inventory.approve_purchase_orders`, `inventory.grn.view`, `inventory.grn.update`, `inventory.grn.approve`, `inventory.grn.reject`, `inventory.direct_orders.view`, `inventory.stock_takes.create`, `inventory.manage_stock_takes`, `inventory.expiry.view`, `inventory.dispensing.dispense`, `inventory.reports.view`, `inventory.view_reports` |
| Point of sale | `pos.manage`, `pos.sales.view`, `pos.sales.create`, `pos.sales.receive_payment`, `pos.shifts.view`, `pos.shifts.manage`, `pos.terminals.view`, `pos.terminals.manage` |
| Dialysis | `dialysis.manage`, `dialysis.setup`, `dialysis.patients.view`, `dialysis.patients.update`, `dialysis.patients.delete`, `dialysis.schedules.view`, `dialysis.schedules.update`, `dialysis.schedules.cancel`, `dialysis.sessions.view`, `dialysis.sessions.create`, `dialysis.sessions.update`, `dialysis.sessions.start`, `dialysis.sessions.record_assessment`, `dialysis.sessions.complete`, `dialysis.sessions.cancel`, `dialysis.deposits.manage`, `dialysis.reports.view` |
| Oncology | `oncology.view`, `oncology.screening.record`, `oncology.diagnosis.write`, `oncology.staging.write`, `oncology.mdt.decide`, `oncology.plan.approve`, `oncology.cycle.open`, `oncology.cytotoxic.handle`, `oncology.toxicity.write`, `oncology.benefit.view` |
| Psychiatry | `psychiatry.access`, `psychiatry.mental_health.access`, `psychiatry.rehabilitation.access`, `psychiatry.therapeutic.access`, `psychiatry.record.view`, `psychiatry.risk.view`, `psychiatry.therapy.view`, `psychiatry.inpatient.edit`, `psychiatry.legal.decide`, `psychiatry.referral.manage`, `psychiatry.reports.view`, `psychiatry.settings.manage`, `psychiatry.access_log.view`, `psychiatry.break_glass.review` |
| Morgue | `morgue.view`, `morgue.manage`, `morgue.admit`, `morgue.intake.request`, `morgue.storage.manage`, `morgue.procedure.record`, `morgue.release.raise`, `morgue.release.approve`, `morgue.unclaimed.manage`, `morgue.charges.view`, `morgue.charges.manage`, `morgue.charges.waive`, `morgue.rates.manage` |
| Users | `users.manage`, `users.view`, `users.create`, `users.update`, `users.users.view`, `users.users.create`, `users.roles.view`, `users.assign_roles`, `users.sync_permissions`, `roles.view` |
| Setup | `settings.manage`, `settings.view`, `settings.edit`, `settings.manage_locations`, `organizations.view`, `regions.view`, `facilities.view`, `departments.view`, `hierarchy.view`, `audit.view` |
| Reports and SMS | `reports.view`, `reports.view_financial_reports`, `sms.manage` |

### 21.3 Suggested roles

These are starting points. Adjust them to how the client divides work.

| Role | Core permissions |
|---|---|
| Receptionist | `dashboard.view`, `reception.manage`, `reception.patients.view`, `reception.patients.create`, `reception.visits.view`, `reception.queue.manage`, `reception.appointments.view` |
| Triage nurse | `dashboard.view`, `triage.view`, `triage.perform` |
| Doctor / clinician | `dashboard.view`, `evaluation.manage`, `clinical_notes.sign`, `procedures.request`, `prescriptions.view`; add `inpatient.admissions.view` and theatre booking permissions if they admit or operate |
| Laboratory technologist | `lab.tests.view`, `lab.samples.view`, `lab.samples.collect`, `lab.samples.receive`, `lab.samples.process`, `lab.results.view`, `investigations.record_results`; add `lab.results.verify` and `investigations.verify_results` for the person who approves results |
| Pharmacist | `pharmacy.dispensing.view`, `pharmacy.dispensing.dispense`, `pharmacy.prescriptions.view`, `pharmacy.prescriptions.fill`, `inventory.stock.view`; add `pos.manage` and `pos.sales.create` for counter sales |
| Cashier | `finance.manage`, `finance.process_payments`, `finance.payments.view`, `finance.invoices.view`, `finance.accounts.view` |
| Billing / claims officer | `finance.manage`, `finance.invoices.view`, `insurance.view`, `insurance_claims.view`, `insurance_claims.submit`, `insurance_claims.resubmit` |
| Finance manager | The cashier and billing permissions plus `credit_notes.approve`, `waivers.approve`, `invoices.void`, `finance.view_financials`, `reports.view`, `reports.view_financial_reports` |
| Ward nurse | `inpatient.admissions.view`, `inpatient.beds.view`, `inpatient.admit_patients`, `triage.perform` |
| Inpatient manager | The ward nurse permissions plus `inpatient.discharge_patients`, `inpatient.manage_wards`, `wards.view` |
| Theatre nurse | `theatre.manage`, `theatre.checkin`, `theatre.schedule.index`, `theatre.wardcheck.nurse`, `theatre.operations.view` |
| Storekeeper | `inventory.manage`, `inventory.items.view`, `inventory.stock.view`, `inventory.requisitions.view`, `inventory.requisitions.create`, `inventory.grn.view`, `inventory.transfer_stock` |
| Procurement officer | The storekeeper permissions plus `inventory.suppliers.view`, `inventory.suppliers.create`, `inventory.purchase_orders.view`, `inventory.purchase_orders.create`; add `inventory.purchase_orders.approve` and `inventory.grn.approve` for the approver |
| System administrator | `users.manage` and the users permissions, `settings.manage`, `settings.view`, `settings.edit`, `settings.manage_locations`, `organizations.view`, `regions.view`, `facilities.view`, `departments.view`, `insurance.view`, `audit.view`, `reports.view` |

Good practice: keep approval permissions (credit notes, waivers, purchase orders, GRNs, morgue releases) with a different person from the one who raises the item.

---

## 22. Installing and deploying the application

This section is for Afya Analytics implementation and support staff. Client staff do not need it.

### 22.1 Requirements

- Node.js version 22 or later on the build machine.
- The backend services for the client must be running and reachable from users' browsers: core, reception, patient evaluation, inventory, inpatient, theatre, dialysis, finance, reports, oncology, morgue, integration and psychiatry.
- A web server (nginx) to serve the built files.

### 22.2 Environment configuration (the `.env` file)

Copy `.env.example` to `.env` and set one address per backend service. Every address ends in `/api/v1`.

| Variable | Service | Local default port |
|---|---|---|
| `CORE_SERVICE_URL` | Core: sign-in, users, organizations, settings, dashboards | 8001 |
| `FINANCE_SERVICE_URL` | Finance: billing, Cash Workbench, invoices | 8002 |
| `PATIENT_EVALUATION_SERVICE_URL` | Outpatient clinical work, lab, pharmacy, SHA | 8003 |
| `RECEPTION_SERVICE_URL` | Patients, appointments, visits, queues | 8004 |
| `INVENTORY_SERVICE_URL` | Stock, stores, procurement, POS | 8005 |
| `THEATRE_SERVICE_URL` | Theatre | 8007 |
| `INPATIENT_SERVICE_URL` | Wards and admissions | 8008 |
| `REPORTS_SERVICE_URL` | Reports | 8009 |
| `DIALYSIS_SERVICE_URL` | Dialysis | 8010 |
| `ONCOLOGY_SERVICE_URL` | Oncology | 8011 |
| `MORGUE_SERVICE_URL` | Morgue | 8012 |
| `INTEGRATION_SERVICE_URL` | Per-organization integration switches | 8013 |
| `PSYCHIATRY_SERVICE_URL` | Psychiatry | 8014 |

Important:
- **Every variable must be set.** There is no built-in fallback address. A missing variable means that service's requests fail, by design, so that a deployment never talks to the wrong client's servers by accident.
- The addresses are read when the application is built. After changing `.env`, rebuild and redeploy; restarting the web server alone is not enough.
- For a client-hosted or on-premise installation, the addresses must be reachable from every user's browser, not only from the server.

Optional variables:

| Variable | Use |
|---|---|
| `NUXT_PUBLIC_REALTIME_ENABLED` | `true` turns on live updates. Off by default; screens then refresh by polling |
| `NUXT_PUBLIC_REVERB_APP_KEY`, `NUXT_PUBLIC_REVERB_PORT`, `NUXT_PUBLIC_REVERB_SCHEME` | Connection details for the live-update server, which runs on the core service. The port defaults to 8080 and the scheme to http |
| `NUXT_PUBLIC_REVERB_HOST` | Leave unset. It defaults to the core service's host. Set it only if the live-update server runs somewhere else |
| `NUXT_PUBLIC_MAINTENANCE_ENABLED`, `_START`, `_END`, `_NOTICE_AFTER_MINUTES` | Build-time defaults for a maintenance window (section 22.6) |
| `FRONTEND_DIR`, `NGINX_USER` | Used by the deploy script: the directory to deploy into and the web server's user |

### 22.3 Running locally for development or a demo

```bash
cp .env.example .env     # then edit the service addresses
npm install
npm run dev              # opens on http://localhost:3000
```

The sign-in page loads even when no backend is running, but signing in needs the core service. If sign-in shows "We couldn't complete this right now", the core service address is wrong or the service is not running.

### 22.4 Building for production

```bash
npm run build      # or: npm run generate
npm run preview    # optional: preview the production build locally
```

The build produces a static site in `.output/public`. The web server must send every unknown path to `index.html`, so that refreshing a deep page such as `/reception/patients` does not return "404 Not Found". In nginx:

```nginx
location / {
    try_files $uri $uri/ /index.html;
}
```

A container image can be built with `Dockerfile.eks`, which builds the site and serves it with nginx on port 8080 as a non-root user. The service addresses are passed as build arguments.

### 22.5 The deploy script

`./deploy.sh` builds and publishes the current code on a server. It needs `.env` with `FRONTEND_DIR` and `NGINX_USER`.

- Run it as the deploy user, not with `sudo`. It asks for elevated rights only for the web server steps.
- Only one deployment runs at a time. A second attempt stops with "Another deploy is already running".
- If the same version is already live and nothing has changed, it stops with "already live — nothing to do".
- It reports whether live updates are on and whether the live-update server answers. This is information only; the deployment continues either way.
- `./deploy.sh rollback` puts the previous build back. "No previous build to roll back to" means there is nothing to restore.

### 22.6 Scheduling a maintenance window

The window is read from `maintenance.json` in the served site root, so it can be changed on a live site without rebuilding:

```json
{ "enabled": true, "start": "2026-10-31T22:00:00+03:00", "end": "2026-11-01T00:00:00+03:00", "noticeAfterMinutes": 30 }
```

- Set `"enabled": true` to show the notice and apply the lock; `false` cancels everything.
- To extend a window, change `end`. The file is re-read every minute and again at the end time.
- Always keep the `+03:00` offset. Times are East Africa Time and are taken from the server clock, so a wrong clock on a user's computer changes nothing.
- The next deployment resets the file to the committed version, so commit the final values too.

### 22.7 Creating a new client (organization)

A new organization is created on the server by Afya Analytics, together with its permission list, its standard roles and its default settings. After that:

1. Sign in as the organization's first administrator.
2. Follow section 3 in order.
3. Optionally load staff in bulk (section 3.25).

If roles or permissions look incomplete for one organization ("No permissions found. Ensure the permissions seeder has run"), the permission list needs to be re-synchronised on the server for that organization.

### 22.8 Technical troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Sign-in shows "We couldn't complete this right now" | Core service unreachable | Check `CORE_SERVICE_URL`, that the service is running, and that the browser can reach it |
| One module shows errors, the rest work | That module's service is down or its address is wrong | Check the address in `.env` and the service's health at `/core/setup/modules` |
| Refreshing a page gives "404 Not Found" | The web server is not sending unknown paths to `index.html` | Add the `try_files` rule (22.4) |
| "403 Forbidden" from the web server on the whole site | File ownership or web server configuration | Check the site directory is owned by the web server user and the root path is right |
| Changed `.env` but nothing changed | The addresses are built in | Rebuild and redeploy |
| Screens do not update live | Live updates are off or the live-update server is not reachable | Expected when off: screens poll instead. To enable, set the realtime variables and start the server |
| The maintenance page does not lift | `maintenance.json` still enabled or the end time is wrong | Correct the file |

---

## 23. Setup checklists by type of client

Use these as a shortlist of what to configure for each kind of client. Section numbers refer to the detailed instructions.

### 23.1 Outpatient clinic (no pharmacy stock, no admissions)

- Organization, one region, one facility, departments (3.2)
- Switch off sections not used: In-Patient, Theatre, Oncology, Psychiatry, Renal/Dialysis, MCH if not offered (3.8)
- Registration form fields (3.8)
- Procedure categories; consultation and procedure services with cash and insurance prices (3.5)
- Destinations: Triage (auto check-in), Consultation, and any procedure room (3.3)
- Insurance companies and schemes, if insured patients are seen (3.6)
- Payment modes (3.7)
- Roles: Receptionist, Triage nurse, Doctor, Cashier, Administrator; then users (3.4)

### 23.2 Clinic with pharmacy and laboratory

Everything in 23.1, plus:
- Inventory: units, categories, products with prices and reorder levels, a Main store and a Pharmacy store, suppliers, opening stock (3.12)
- A Pharmacy destination marked as a pharmacy queue and connected to the Pharmacy store (3.3)
- A point of sale for over-the-counter sales, with cashiers authorised (3.13)
- A Laboratory destination marked as a lab queue (3.3)
- Laboratory setup: categories, units, sample types, tests, reference ranges, critical values (18)
- Roles: Pharmacist, Laboratory technologist, Storekeeper

### 23.3 Hospital with wards and theatre

Everything in 23.2, plus:
- Inpatient: bed types, admission types with deposits, discharge types, wards, beds, recurring and one-off fees (3.14)
- Ward stores, so drugs and consumables can be issued to wards (3.12)
- Theatre: theatre types, operation theatres, theatre deposits; a theatre store (3.15)
- Service charges and rebates (3.7)
- Radiology destination and imaging procedures (3.3, 3.5)
- Approval levels for requisitions and purchase orders (3.12)
- Roles: Ward nurse, Inpatient manager, Theatre nurse, Procurement officer, Finance manager, Billing officer

### 23.4 Multi-branch group

Everything that applies above, for each branch, plus:
- One region per area and one facility per branch; mark the head office as Main (3.2)
- Destinations, stores, points of sale, wards, theatres and procedure prices for each facility
- Assign each user to the right facilities; decide per role whether it has **Concurrent multi-facility viewing** (3.4)
- Check that the SHA FR Code is set on each branch that claims (3.2)

### 23.5 Maternity and child health facility

- The outpatient set from 23.1
- Keep the MCH section on (3.8)
- A triage destination with the Vitals action, so patients can be put on the ANC, PNC and CWC enrollment queues (3.20)
- A maternity ward with **Provides maternity care** ticked, if deliveries are done (3.14)
- MOH registers under Reports (19)

### 23.6 Dialysis unit

- The outpatient set from 23.1
- Dialysis setup in order: plants, ward types, wards, machines, chairs, vascular access types, coagulation types, deposits (3.16)
- Dialysis permissions for nurses and the unit manager (21)
- SHA setup if sessions are claimed (3.21)

### 23.7 Facility with a mortuary

- Switch the Morgue section on (3.8)
- Add fridges and spaces (3.17)
- Morgue permissions, with release approval held by a different person from the one raising releases (21)
- Mortuary charges are collected at the Cash Workbench using the Morgue toggle (7.2)

### 23.8 Mental health or rehabilitation facility

- The outpatient set from 23.1
- Psychiatry settings: forms, patient-facing wording, crisis line, safety gates, timings (3.18)
- Psychiatry, Rehabilitation and Therapeutic destinations as needed (3.3)
- Psychiatric wards with a Psychiatry Service set (3.14)
- Service permissions per clinician, and a reviewer for emergency access (21)

### 23.9 Any client using SHA

- All eight steps in section 3.21
- HealthID and a fingerprint reader on the workstations that capture consent (3.22)
- ICD-11 diagnosis recorded on every SHA visit (6.5)

---

## 24. Frequently asked questions and troubleshooting

### Setup and configuration

**In what order should I set up a new client?**
Organization, regions and facilities, departments, sidebar sections and settings, procedures and prices, destinations, insurance, payment modes, inventory, points of sale, then the modules the client uses, and finally roles and users. Section 3.1 has the full list and section 23 has a checklist per type of client.

**A setting in General Settings is greyed out with "These settings are locked and cannot be changed here."**
Most settings are locked on purpose and are changed by Afya Analytics support. Only the billing-bypass switches and the SHA settings can be changed on screen (section 3.8). Send support the setting key from section 20 and the value you want.

**How do I hide a module the client does not use?**
`Setup > General Settings > Core`, under "Side menu", switch the section off (section 3.8). Its setup screens are then greyed out.

**The Morgue menu is missing for everyone.**
The Morgue section is off by default. Switch it on in `General Settings > Core`, then grant the morgue permissions.

**I created a destination but it does not appear under OutPatient.**
Check that it is active, belongs to the user's facility, is not ticked **Hidden from normal queues**, and that its type is a room or department. Facility units such as Inpatient, Theatre, Dialysis and MCH have their own sidebar sections and are not listed under OutPatient (section 3.3).

**Can I create Theatre or Dialysis as a destination?**
No. They are separate modules with their own setup (sections 3.15 and 3.16).

**How do I make every patient go to Triage first?**
Tick **Auto checkin** on the Triage destination. It is then pre-selected at every check-in.

**The clinical screen for a queue is missing a tab, such as Prescribe.**
Edit the destination and add that tab under **Treatment Actions** (section 3.3).

**Lab Settings says no destination is configured as the Laboratory.**
Create or edit the laboratory destination and tick **Is a lab queue**.

**A pharmacist or lab technologist cannot see their queue.**
Mark the destination as a pharmacy queue or lab queue and give the user `pharmacy.dispensing.view` or `lab.tests.view`. They do not need the general clinical permission.

**How do I set a different price for one insurer?**
Open the scheme's **Drugs & Procedure Exclusions** and set a **Scheme Price**, or open the company's **Drugs & Procedure Pricing** and set a company price for all its schemes (section 3.6).

**An insured patient was charged the cash price.**
No scheme, company or insurance price exists for the item, so the cash price was used as a last resort; or the visit has no scheme on it. Set the insurance price and check the visit's payment details.

**How do I load many prices, exclusions, categories or reference ranges at once?**
Use the bulk tools: **Upload Excel** on scheme exclusions and prices, **Bulk Import Categories (CSV)** on procedure categories, **Bulk Update Prices (CSV)** on products, and **Bulk Import (CSV)** on reference ranges and critical values. Each has a template to download.

**The logo is not showing on receipts.**
Upload a square **Logo** on the organization (section 3.2). Thermal receipts use the logo, not the letterhead, and the receipt logo setting must be on (`no_pos.receipt_show_logo`).

**Can I upload a PDF letterhead?**
No. Upload an image.

**The role permission screen says "No permissions found. Ensure the permissions seeder has run."**
The organization's permission list is missing. Support re-synchronises it on the server.

**I created a user but did not copy the password.**
The password is shown only once. Reset it under the user's **Credentials** tab (section 2.5).

**A new user can sign in but sees no data.**
They are not assigned to a facility. Tick the facility under **Authorized Regions & Facilities**.

**A user should see data from all branches at once.**
Switch on **Concurrent multi-facility viewing** on their role and assign them to those facilities.

**A cashier cannot see a point of sale.**
The point of sale must be Active, and the cashier must be authorised for it under **Authorized POS Stations** and hold `pos.sales.create` (section 3.13).

**A doctor cannot find a drug when prescribing.**
The product must exist and be active, have stock in the store connected to the pharmacy, and the store must be open if opening hours are restricted (section 3.12).

**How do I enter opening stock?**
`Inventory > Stores > List Stores > Manage > Add Items` for each store (section 3.12).

**Prescriptions are dispensing the wrong number of tablets.**
Check the product's **Dispensable Amount** (tablets inside one sold unit), **Pack Size** and **Sold As**. Use the Packaging helper on the product form.

**Who can approve requisitions and purchase orders?**
The users assigned to approval levels at `Setup > Inventory Settings > Approval Levels`. A higher weight means more authority.

**How are IP numbers generated?**
Automatically by default (`inpatient.auto_generate_inpatient_no`), with an optional prefix (`inpatient.inpatient_no_prefix`).

**Can two patients share a bed?**
Only if over-booking is allowed (`inpatient.allow_overbooking_of_wards`).

**Dialysis sessions cannot be scheduled.**
At least one active machine and one chair must exist, and the chair must be linked to a ward and machine (section 3.16).

**What does a client need before SHA claims work?**
The eight steps in section 3.21: SHA switched on, an SHA scheme, the SHA settings, the facility FR code, practitioner ids, coded products and procedures, the HealthID workstation, and claim permissions.

**The page says it could not reach HealthID or the fingerprint reader.**
The HealthID application or the SecuGen service is not installed, not running or not signed in on that computer (section 3.22).

**Can each client have its own colours and logo?**
Yes. Colours are set at `Setup > UI Colors` and the logo and letterhead on the organization (sections 3.2 and 3.11).

**The currency symbol is wrong.**
It comes from `inventory.currency` and `inventory.currency_symbol`. Ask support to change them.

**How do I see who changed a record?**
`Setup > Sudo > Audit Trail` (permission `audit.view`).

### Access and sign-in

**I cannot sign in and see "We couldn't complete this right now".**
The request did not reach the server. Check your internet connection and try again. If everyone at your facility is affected, contact support.

**I forgot my password.**
Click **Forgot password?** on the sign-in page and follow section 2.3. An administrator can also reset it (2.5).

**My account setup link has expired.**
Setup links last 8 hours. Click **Request new link** on the expired page, or ask your administrator.

**A menu or button I need is missing.**
Your role does not include the permission for it, or you are not assigned to the facility. Ask your administrator to check your roles, special permissions and assigned facilities (3.4), then sign out and back in.

**I see a "403" or "not authorised" page.**
You opened a page your role does not allow. Ask your administrator for the permission.

**I am working in the wrong facility.**
The facility badge in the top bar shows your current facility. Sign out and sign in again, choosing the right facility.

### Reception

**The system says "Patient already has an active visit".**
The earlier visit was never checked out. Choose **Check out** to close it, **Open full visit** to continue it, or **Create new visit** (5.5).

**The ID number field is disabled.**
The patient is under 18 from their date of birth, so an ID number is not required.

**A patient was sent to the wrong queue.**
`Reception > Manage Queue`, then **Change Destination** on the visit (5.6). Clinicians can also use **CHECKIN ELSEWHERE**.

**How do I add a second queue without removing the first?**
In **Change Destination** use **Add Destination** or **Checkin elsewhere**.

### Outpatient, lab and pharmacy

**A queue is missing from the OutPatient menu.**
The destination is not set up, is inactive, or belongs to another facility; or the user lacks the permission (3.3).

**The patient is not in my queue.**
Check the date filter, the facility filter and that the visit was routed to your destination. Click **Refresh**.

**A lab test shows "Procedure not paid for yet".**
A cash patient must pay at the cashier before the test proceeds.

**I cannot find a drug when prescribing.**
The drug must exist as a product, be in stock in the store you selected, and that store must be open. "Store closed" means it is outside its opening hours.

**A drug shows "Product not covered by patient scheme".**
The patient's insurance scheme excludes it. The patient pays cash for that item.

**I cannot dispense a prescription.**
It must first be processed and paid for (or be on insurance). Check its payment status under the ordered drugs (6.9).

### Finance

**The cashier cannot see the patient's bill.**
The item may have no price, may already be paid or invoiced, or the visit is on insurance (look in the Insurance Workbench). For theatre or mortuary charges, switch on the **Theatre** or **Morgue** toggle in the Cash Workbench.

**I cannot bill an item because it is 0 Ksh.**
Set its price in the procedure or product setup first.

**How do I refund or reverse a payment?**
Issue a credit note on the receipt and have it approved (7.4).

**I cannot dispatch several invoices together.**
Invoices in one dispatch must belong to the same insurance scheme.

**I closed the day by mistake.**
Day Close cannot be undone. Contact support.

### Inpatient and theatre

**I cannot admit the patient.**
The admission request must be authorized first, with the deposit paid or waived (8.1).

**Beds Management shows every bed as available.**
There are no active admissions. This reflects real admissions.

**I cannot change a charge on the charge sheet.**
A charge that has been invoiced must be corrected on its invoice in Finance.

**The Schedule button is missing on a theatre booking.**
The booking must be Approved or Authorized first (9.1).

**"This timeslot is already booked."**
Choose a different time or operating room.

### Inventory and POS

**Stock did not increase after receiving goods.**
Stock updates only when the GRN is approved with **Approve & Update Stock** (10.3).

**I cannot save because of batch or expiry errors.**
Products that track batch, expiry or serial numbers need those values on every line.

**I cannot open the POS terminal.**
The station must be set up and active, linked to a store, and you must be authorised for it under **Authorized POS Stations** (11.1). Open a shift before selling.

**My shift shows a variance.**
The counted cash differs from the expected cash. Recount, and record the reason in the closing notes.

### SHA

**The SHA menu is missing.**
SHA integration is switched off for your organization, or you lack the claims permissions.

**I cannot submit a SHA claim.**
Open the claim and work through its checklist (section 17). The commonest causes are a missing ICD-11 diagnosis, missing laterality, no bill lines, or a pending preauthorisation.

**A claim shows "Time barred".**
It was not submitted within 7 days. Contact your SHA claims officer.

### Getting more help

If this guide does not solve the problem, contact your organization's system administrator first. They can check your permissions and your facility's setup. For faults affecting everyone, or anything that looks like wrong data, your administrator should contact Afya Analytics support with the page address, the patient or record number, the time it happened and a screenshot.

---

## 25. Glossary

| Term | Meaning |
|---|---|
| Admission request | A clinician's request to admit a patient, which must be authorized before a bed is given |
| ANC / PNC / CWC | Antenatal care / postnatal care / child welfare clinic |
| Capitation | A fixed amount an insurer covers for a visit |
| Charge sheet | The day-by-day list of charges on an admission |
| Check-in | Starting a visit and sending the patient to one or more queues |
| Checkout | Ending a patient's time in a queue or closing the visit |
| Copay | The part of an insured bill the patient pays |
| Credit note | A document that reverses all or part of a receipt or invoice |
| Destination | A place a patient is sent during a visit; each one is a queue |
| Dispatch | Sending a batch of invoices to an insurer |
| EDD / LMP | Expected date of delivery / last menstrual period |
| Facility | One branch or site of your organization |
| GRN | Goods Received Note; approving it adds stock |
| ICD-11 | The international diagnosis coding list used for diagnoses and SHA claims |
| IP number | Inpatient number given at admission |
| Knock off | Removing damaged or expired stock |
| LPO / PO | Local purchase order / purchase order |
| MOH registers | Ministry of Health registers such as MOH 405, 510, 511 and 711 |
| NKDA / NKFA | No known drug allergies / no known food allergies |
| Preauthorisation | An insurer's approval required before a service is given |
| Requisition | A request for stock, from a supplier or from another store |
| Scheme | A specific insurance plan under an insurance company |
| SHA | Social Health Authority, Kenya's public health insurer |
| SHR | Shared Health Record, the national patient record |
| TCA | "To come again" — a follow-up instruction |
| Treatment action | A tab on the clinical screen, configured per destination |
| Triage | Assessing urgency from vitals; sets the queue colour |
| Visit | One attendance by a patient, from check-in to checkout |
| Waiver | An approved reduction of a patient's bill |
