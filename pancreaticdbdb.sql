-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Mar 09, 2024 at 02:20 PM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.1.17

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `pancreaticdbdb`
--

-- --------------------------------------------------------

--
-- Table structure for table `plantdata`
--

CREATE TABLE `plantdata` (
  `id` int(11) NOT NULL,
  `plant_name` varchar(255) NOT NULL,
  `scientific_name` varchar(255) DEFAULT NULL,
  `description` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `plantdata`
--

INSERT INTO `plantdata` (`id`, `plant_name`, `scientific_name`, `description`) VALUES
(1, 'Algae', 'Chlorella Vulgaris', 'Chlorella vulgaris is a unicellular green algae commonly found in freshwater environments. It has a spherical shape and is often used as a nutritional supplement due to its high protein content and other beneficial nutrients.'),
(2, 'AmericanLotus', 'Nelumbo Lutea', 'American lotus plants thrive in shallow freshwater habitats such as ponds lakes marshes and slow-moving streams. They prefer muddy or silty substrate and are often found in areas with full sun exposure.'),
(3, 'BananaLilly', 'Nymphoides Aquatica', 'Banana Lily or Nymphoides aquatica is an aquatic perennial plant native to North America. It typically grows in shallow waters of ponds lakes and slow-moving streams.'),
(4, 'Bladderwort', 'Utricularia Vulgaris', 'Bladderwort is an aquatic or semi-aquatic plant found in a wide range of freshwater habitats including ponds lakes streams and marshes. It is known for its ability to thrive in nutrient-poor environments.'),
(5, 'Cattail', 'Typha Latifolia', 'Cattails are ubiquitous wetland plants found in various aquatic habitats including marshes swamps ponds and the edges of lakes and streams. They are widespread and can tolerate a wide range of environmental conditions including fluctuating water levels and varying degrees of salinity.'),
(6, 'Chara', 'Chara Vulgaris', 'Chara algae are multicellular and typically form branching bushy structures that resemble submerged plants. They have a greenish coloration due to the presence of chlorophyll which allows them to photosynthesize. '),
(7, 'CommonReed', 'Phragmites Australis', 'Common Reed is a tall perennial grass species that is widely distributed across temperate and tropical regions of the world. It typically grows in wetland habitats such as marshes swamps and along the edges of lakes ponds and rivers.'),
(8, 'Coontail', 'Ceratophyllum Demersum', 'Coontail is a submerged aquatic plant found in ponds lakes streams and other freshwater habitats. It can grow in a wide range of conditions from shallow to deep water and is often found in nutrient-rich environments.'),
(9, 'Duckweed', 'Lemna Spp', 'Duckweed plants are very small ranging from a few millimeters to a centimeter or so in size. They consist of one to several oval-shaped or round leaves which are typically flat and float on the waters surface. '),
(10, 'Eelgrass', 'Zostera Marina', 'Common Eelgrass is a marine angiosperm found in shallow coastal waters estuaries and bays around the world. It grows in soft sediments such as sand or mud and is often found in areas with calm water and moderate salinity.'),
(11, 'Fanwort', 'Cabomba Caroliniana', 'Fanwort or Cabomba caroliniana is a submerged aquatic plant native to the southeastern United States and parts of South America. It typically grows in slow-moving or still waters such as ponds lakes and streams.'),
(12, 'GaintBulrush', 'Schoenoplectus Californicus', 'Giant Bulrush is a perennial rhizomatous aquatic plant native to North America. Its commonly found in freshwater habitats such as marshes swamps and the margins of lakes and ponds.'),
(13, 'Hydrilla', 'Hydrilla Verticillata', 'Hydrilla is a submerged aquatic plant native to Asia but it has been introduced to many parts of the world. It thrives in a wide range of aquatic environments including lakes ponds rivers and canals. Hydrilla can grow in both freshwater and brackish water habitats.'),
(14, 'Hygrophila', 'Hygrophila Polysperma', 'Native to Southeast Asia particularly countries like India Thailand Malaysia and Indonesia Hygrophila polysperma is found in a variety of aquatic environments including streams ponds and marshes. It is often cultivated in aquariums worldwide due to its attractive appearance and ease of care.'),
(15, 'PennyWort', 'Hydrocotyle Ranunculoides', 'Pennywort species are generally found in wet or marshy areas including the edges of ponds lakes streams and other bodies of water. They can also grow in moist soil.'),
(16, 'Pondweed', 'Potamogeton Nodosus', 'Pondweed species are found in a variety of aquatic habitats including ponds lakes slow-moving streams and marshes. They are widespread across temperate and subtropical regions worldwide.'),
(17, 'Sawgrass', 'Cladium Mariscus', 'Sawgrass is a perennial rhizomatous plant that is commonly found in wetland habitats such as marshes swamps and along the edges of ponds and streams. It often forms dense stands in these environments.'),
(18, 'SouthernNaiad', 'Najas Guadalupensis', 'Southern Naiad is a submerged aquatic plant that is native to the southeastern United States Central America and northern South America. It typically grows in quiet or slow-moving freshwater habitats such as ponds lakes marshes and streams.'),
(19, 'SpatterDock', 'Nuphar Lutea', 'Spatterdock or Nuphar lutea is a perennial aquatic plant native to Europe Asia and North America. It typically grows in shallow waters of ponds lakes slow-moving streams and marshes.'),
(20, 'WaterFern', 'Azolla', 'Water ferns like Azolla are small floating aquatic ferns that inhabit still or slow-moving bodies of water such as ponds lakes and marshes. They are found worldwide in temperate and tropical regions.'),
(21, 'WaterHyacinth', 'Eichhornia Crassipes', 'Water hyacinth is native to South America but has become naturalized and invasive in many tropical and subtropical regions around the world. It thrives in freshwater habitats such as ponds lakes rivers and marshes with slow-moving or stagnant water.'),
(22, 'WaterLettuce', 'Pistia Stratiotes', 'Water Lettuce is a free-floating aquatic plant native to tropical and subtropical regions around the world. It thrives in slow-moving or stagnant freshwater habitats such as ponds lakes marshes and slow-moving streams.'),
(23, 'WhiteWaterLiliy', 'Nymphaea Odorata', 'White Water Lily or Nymphaea odorata is a perennial aquatic plant native to North America. It thrives in shallow waters of ponds lakes and slow-moving streams where it can root in the mud at the bottom.');

-- --------------------------------------------------------

--
-- Table structure for table `userdata`
--

CREATE TABLE `userdata` (
  `Named` varchar(50) DEFAULT NULL,
  `Email` varchar(50) DEFAULT NULL,
  `Pswd` varchar(50) DEFAULT NULL,
  `Phone` varchar(50) DEFAULT NULL,
  `Addr` varchar(4000) DEFAULT NULL,
  `Dob` varchar(50) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;

--
-- Dumping data for table `userdata`
--

INSERT INTO `userdata` (`Named`, `Email`, `Pswd`, `Phone`, `Addr`, `Dob`) VALUES
('Hrushitha', 'test@gmail.com', 'Qazwsx@123', '9090909090', 'Demo', '02/04/2024'),
('Bhoomika', 'bhoomi@gmail.com', 'Bhoomika', '2383838907', 'Hebbal\n', '28/12/2004'),
('Tarun', 'tarun@gmail.com', 'qazwsx', '9876543211', 'Vani Vilas Water Supply - Mysore KRS Road Yadavagiri', '03/05/2024');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `plantdata`
--
ALTER TABLE `plantdata`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `plantdata`
--
ALTER TABLE `plantdata`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=24;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
