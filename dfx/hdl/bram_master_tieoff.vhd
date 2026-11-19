----------------------------------------------------------------------------------
-- bram_master_tieoff: BRAM master port that never accesses anything (the CNN and the SNN have no feature-map port, but every module of a slot needs the same boundary)
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;

entity bram_master_tieoff is
  port (
    fm_addr : out std_logic_vector(31 downto 0);
    fm_en   : out std_logic;
    fm_we   : out std_logic_vector(1 downto 0);
    fm_din  : out std_logic_vector(15 downto 0);
    fm_dout : in  std_logic_vector(15 downto 0);
    fm_clk  : out std_logic;
    fm_rst  : out std_logic
  );

end entity bram_master_tieoff;

architecture Behavioral of bram_master_tieoff is
  -- Vivado infers the buses from these attributes (module reference in a block design).
  attribute X_INTERFACE_INFO      : string;
  attribute X_INTERFACE_MODE      : string;
  attribute X_INTERFACE_PARAMETER : string;
  attribute X_INTERFACE_INFO of fm_addr : signal is "xilinx.com:interface:bram:1.0 feature_map ADDR";
  attribute X_INTERFACE_MODE of fm_addr : signal is "master";
  attribute X_INTERFACE_PARAMETER of fm_addr : signal is "MASTER_TYPE BRAM_CTRL, MEM_ADDRESS_MODE BYTE_ADDRESS, MEM_ECC NONE, MEM_SIZE 1280, MEM_WIDTH 16, READ_LATENCY 1";
  attribute X_INTERFACE_INFO of fm_en   : signal is "xilinx.com:interface:bram:1.0 feature_map EN";
  attribute X_INTERFACE_INFO of fm_we   : signal is "xilinx.com:interface:bram:1.0 feature_map WE";
  attribute X_INTERFACE_INFO of fm_din  : signal is "xilinx.com:interface:bram:1.0 feature_map DIN";
  attribute X_INTERFACE_INFO of fm_dout : signal is "xilinx.com:interface:bram:1.0 feature_map DOUT";
  attribute X_INTERFACE_INFO of fm_clk  : signal is "xilinx.com:interface:bram:1.0 feature_map CLK";
  attribute X_INTERFACE_INFO of fm_rst  : signal is "xilinx.com:interface:bram:1.0 feature_map RST";
begin
  fm_addr <= (others => '0');
  fm_en   <= '0';
  fm_we   <= "00";
  fm_din  <= (others => '0');
  fm_clk  <= '0';
  fm_rst  <= '0';
end architecture Behavioral;