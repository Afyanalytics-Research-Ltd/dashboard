--
-- PostgreSQL database dump
--

\restrict PFn8I7qWa1m4tcBseTmswVwYnZb6G0lllhVriDqShcCAbEIBvBTLUCKWrzpvibS

-- Dumped from database version 16.11 (Debian 16.11-1.pgdg13+1)
-- Dumped by pg_dump version 16.11 (Debian 16.11-1.pgdg13+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: clinic; Type: SCHEMA; Schema: -; Owner: postgres
--

CREATE SCHEMA clinic;


ALTER SCHEMA clinic OWNER TO postgres;

--
-- Name: sex; Type: TYPE; Schema: clinic; Owner: postgres
--

CREATE TYPE clinic.sex AS ENUM (
    'male',
    'female'
);


ALTER TYPE clinic.sex OWNER TO postgres;

--
-- Name: touch(); Type: FUNCTION; Schema: clinic; Owner: postgres
--

CREATE FUNCTION clinic.touch() RETURNS trigger
    LANGUAGE plpgsql
    AS $$ BEGIN NEW.notes := 'x;y'; RETURN NEW; END; $$;


ALTER FUNCTION clinic.touch() OWNER TO postgres;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: patients; Type: TABLE; Schema: clinic; Owner: postgres
--

CREATE TABLE clinic.patients (
    id integer NOT NULL,
    name text NOT NULL,
    sex clinic.sex,
    dob date,
    tags text[],
    extra jsonb,
    registered timestamp with time zone DEFAULT now(),
    balance numeric(12,2)
);


ALTER TABLE clinic.patients OWNER TO postgres;

--
-- Name: TABLE patients; Type: COMMENT; Schema: clinic; Owner: postgres
--

COMMENT ON TABLE clinic.patients IS 'Registered patients';


--
-- Name: COLUMN patients.dob; Type: COMMENT; Schema: clinic; Owner: postgres
--

COMMENT ON COLUMN clinic.patients.dob IS 'Date of birth';


--
-- Name: patients_id_seq; Type: SEQUENCE; Schema: clinic; Owner: postgres
--

CREATE SEQUENCE clinic.patients_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE clinic.patients_id_seq OWNER TO postgres;

--
-- Name: patients_id_seq; Type: SEQUENCE OWNED BY; Schema: clinic; Owner: postgres
--

ALTER SEQUENCE clinic.patients_id_seq OWNED BY clinic.patients.id;


--
-- Name: visits; Type: TABLE; Schema: clinic; Owner: postgres
--

CREATE TABLE clinic.visits (
    id bigint NOT NULL,
    patient_id integer NOT NULL,
    visited_at timestamp without time zone,
    notes text,
    paid boolean
);


ALTER TABLE clinic.visits OWNER TO postgres;

--
-- Name: visits_id_seq; Type: SEQUENCE; Schema: clinic; Owner: postgres
--

CREATE SEQUENCE clinic.visits_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE clinic.visits_id_seq OWNER TO postgres;

--
-- Name: visits_id_seq; Type: SEQUENCE OWNED BY; Schema: clinic; Owner: postgres
--

ALTER SEQUENCE clinic.visits_id_seq OWNED BY clinic.visits.id;


--
-- Name: lookup; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.lookup (
    code character varying(10) NOT NULL,
    label text
);


ALTER TABLE public.lookup OWNER TO postgres;

--
-- Name: patients id; Type: DEFAULT; Schema: clinic; Owner: postgres
--

ALTER TABLE ONLY clinic.patients ALTER COLUMN id SET DEFAULT nextval('clinic.patients_id_seq'::regclass);


--
-- Name: visits id; Type: DEFAULT; Schema: clinic; Owner: postgres
--

ALTER TABLE ONLY clinic.visits ALTER COLUMN id SET DEFAULT nextval('clinic.visits_id_seq'::regclass);


--
-- Data for Name: patients; Type: TABLE DATA; Schema: clinic; Owner: postgres
--

INSERT INTO clinic.patients VALUES (1, 'Amina O''Brien', 'female', '1990-05-01', '{a,b}', '{"k": "v;w"}', '2026-11-17 14:23:28.325402+00', 1250.50);
INSERT INTO clinic.patients VALUES (2, 'Tab	here and back\slash', 'male', NULL, NULL, NULL, '2026-11-17 14:23:28.325402+00', NULL);
INSERT INTO clinic.patients VALUES (3, 'Line
break; with semicolon', NULL, '2001-12-31', '{}', '[]', '2026-11-17 14:23:28.325402+00', 0.00);


--
-- Data for Name: visits; Type: TABLE DATA; Schema: clinic; Owner: postgres
--

INSERT INTO clinic.visits VALUES (1, 2, '2024-01-01 01:00:00', 'visit 1', false);
INSERT INTO clinic.visits VALUES (2, 3, '2024-01-01 02:00:00', 'visit 2', true);
INSERT INTO clinic.visits VALUES (3, 1, '2024-01-01 03:00:00', 'visit 3', false);
INSERT INTO clinic.visits VALUES (4, 2, '2024-01-01 04:00:00', 'visit 4', true);
INSERT INTO clinic.visits VALUES (5, 3, '2024-01-01 05:00:00', 'visit 5', false);


--
-- Data for Name: lookup; Type: TABLE DATA; Schema: public; Owner: postgres
--

INSERT INTO public.lookup VALUES ('A', 'Alpha');
INSERT INTO public.lookup VALUES ('B', NULL);


--
-- Name: patients_id_seq; Type: SEQUENCE SET; Schema: clinic; Owner: postgres
--

SELECT pg_catalog.setval('clinic.patients_id_seq', 3, true);


--
-- Name: visits_id_seq; Type: SEQUENCE SET; Schema: clinic; Owner: postgres
--

SELECT pg_catalog.setval('clinic.visits_id_seq', 5000, true);


--
-- Name: patients patients_pkey; Type: CONSTRAINT; Schema: clinic; Owner: postgres
--

ALTER TABLE ONLY clinic.patients
    ADD CONSTRAINT patients_pkey PRIMARY KEY (id);


--
-- Name: visits visits_pkey; Type: CONSTRAINT; Schema: clinic; Owner: postgres
--

ALTER TABLE ONLY clinic.visits
    ADD CONSTRAINT visits_pkey PRIMARY KEY (id);


--
-- Name: lookup lookup_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lookup
    ADD CONSTRAINT lookup_pkey PRIMARY KEY (code);


--
-- Name: visits visits_patient_id_fkey; Type: FK CONSTRAINT; Schema: clinic; Owner: postgres
--

ALTER TABLE ONLY clinic.visits
    ADD CONSTRAINT visits_patient_id_fkey FOREIGN KEY (patient_id) REFERENCES clinic.patients(id);


--
-- PostgreSQL database dump complete
--

\unrestrict PFn8I7qWa1m4tcBseTmswVwYnZb6G0lllhVriDqShcCAbEIBvBTLUCKWrzpvibS

