-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Server version: 10.4.32-MariaDB

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `clinic`
--

-- --------------------------------------------------------

--
-- Table structure for table `patients`
--

CREATE TABLE `patients` (
  `id` int(11) NOT NULL,
  `patient_no` varchar(250) DEFAULT NULL COMMENT 'Hospital number',
  `name` varchar(100) NOT NULL,
  `dob` date DEFAULT NULL,
  `gender` enum('Male','Female') DEFAULT NULL,
  `balance` decimal(10,2) NOT NULL DEFAULT 0.00,
  `note` text DEFAULT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci COMMENT='Registered patients';

--
-- Dumping data for table `patients`
--

INSERT INTO `patients` (`id`, `patient_no`, `name`, `dob`, `gender`, `balance`, `note`, `created_at`) VALUES
(1, '947/26', 'Hannah Nduta', '0000-00-00', 'Female', 1250.50, 'kagaa', '2026-11-17 05:47:33'),
(2, NULL, 'O\'Brien \"Jr\"', '1990-05-01', 'Male', 0.00, 'line one\nline two; with a semicolon', '2026-11-16 10:44:10'),
(3, '', 'Back\\slash', NULL, NULL, -20.00, '', '2021-09-09 08:50:53');

-- --------------------------------------------------------

CREATE TABLE `visits` (
  `id` bigint(20) UNSIGNED NOT NULL,
  `patient_id` int(11) NOT NULL,
  `visit_date` datetime DEFAULT NULL,
  `amount` decimal(15,2) DEFAULT NULL,
  `paid` tinyint(1) NOT NULL DEFAULT 0
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO `visits` (`id`, `patient_id`, `visit_date`, `amount`, `paid`) VALUES
(10, 1, '2026-11-17 13:03:00', 500.00, 1),
(11, 2, '0000-00-00 00:00:00', NULL, 0),
(12, 1, '2026-11-18 08:00:00', 'n/a', 0);

--
-- Indexes for dumped tables
--

ALTER TABLE `patients`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `patient_no` (`patient_no`);

ALTER TABLE `visits`
  ADD PRIMARY KEY (`id`),
  ADD KEY `patient_id` (`patient_id`);

--
-- AUTO_INCREMENT for dumped tables
--

ALTER TABLE `patients`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

ALTER TABLE `visits`
  ADD CONSTRAINT `visits_ibfk_1` FOREIGN KEY (`patient_id`) REFERENCES `patients` (`id`) ON DELETE CASCADE;
COMMIT;
