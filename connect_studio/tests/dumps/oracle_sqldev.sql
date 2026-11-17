--------------------------------------------------------
--  File created - Monday-November-17-2026
--------------------------------------------------------
--------------------------------------------------------
--  DDL for Table PATIENTS
--------------------------------------------------------

  CREATE TABLE "HR"."PATIENTS"
   (	"ID" NUMBER(10,0),
	"NAME" VARCHAR2(200 BYTE),
	"DOB" DATE,
	"BALANCE" NUMBER(12,2),
	"NOTES" CLOB
   ) SEGMENT CREATION IMMEDIATE
  PCTFREE 10 PCTUSED 40 INITRANS 1 MAXTRANS 255
 NOCOMPRESS LOGGING
  STORAGE(INITIAL 65536 NEXT 1048576 MINEXTENTS 1 MAXEXTENTS 2147483645
  PCTINCREASE 0 FREELISTS 1 FREELIST GROUPS 1
  BUFFER_POOL DEFAULT FLASH_CACHE DEFAULT CELL_FLASH_CACHE DEFAULT)
  TABLESPACE "USERS" ;

   COMMENT ON TABLE "HR"."PATIENTS"  IS 'Registered patients';
--------------------------------------------------------
--  DDL for Table VISITS
--------------------------------------------------------

  CREATE TABLE "HR"."VISITS"
   (	"ID" NUMBER,
	"PATIENT_ID" NUMBER(10,0),
	"VISITED_AT" TIMESTAMP (6)
   ) TABLESPACE "USERS" ;
REM INSERTING into HR.PATIENTS
SET DEFINE OFF;
Insert into HR.PATIENTS (ID,NAME,DOB,BALANCE,NOTES) values ('1','Amina O''Brien',to_date('01-MAY-90','DD-MON-RR'),'1250.5',null);
Insert into HR.PATIENTS (ID,NAME,DOB,BALANCE,NOTES) values ('2','Juma',to_date('2001-12-31 00:00:00','YYYY-MM-DD HH24:MI:SS'),null,'a;b');
REM INSERTING into HR.VISITS
SET DEFINE OFF;
Insert into HR.VISITS (ID,PATIENT_ID,VISITED_AT) values ('1','1',to_timestamp('15-JAN-24 09.30.00.000000000 AM','DD-MON-RR HH.MI.SSXFF AM'));
Insert into HR.VISITS (ID,PATIENT_ID,VISITED_AT) values ('2','2',null);
--------------------------------------------------------
--  Constraints for Table PATIENTS
--------------------------------------------------------

  ALTER TABLE "HR"."PATIENTS" ADD CONSTRAINT "PATIENTS_PK" PRIMARY KEY ("ID")
  USING INDEX PCTFREE 10 INITRANS 2 MAXTRANS 255 COMPUTE STATISTICS
  TABLESPACE "USERS"  ENABLE;
  ALTER TABLE "HR"."PATIENTS" MODIFY ("NAME" NOT NULL ENABLE);
--------------------------------------------------------
--  Ref Constraints for Table VISITS
--------------------------------------------------------

  ALTER TABLE "HR"."VISITS" ADD CONSTRAINT "VISITS_FK1" FOREIGN KEY ("PATIENT_ID")
	  REFERENCES "HR"."PATIENTS" ("ID") ENABLE;
--------------------------------------------------------
--  DDL for Procedure LOG_VISIT
--------------------------------------------------------
set define off;

  CREATE OR REPLACE EDITIONABLE PROCEDURE "HR"."LOG_VISIT" (p_id NUMBER) AS
BEGIN
  INSERT INTO HR.VISITS (ID, PATIENT_ID) VALUES (99, p_id);
  COMMIT;
END;

/
