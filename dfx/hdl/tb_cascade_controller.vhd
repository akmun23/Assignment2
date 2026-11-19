----------------------------------------------------------------------------------
-- tb_cascade_controller: Testbench of cascade_controller with models of the DFX Controller and the networks
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity tb_cascade_controller is
end entity tb_cascade_controller;

architecture sim of tb_cascade_controller is
  signal clk    : std_logic := '0';
  signal resetn : std_logic := '0';

  signal awaddr, araddr : std_logic_vector(7 downto 0) := (others => '0');
  signal awvalid, wvalid, arvalid : std_logic := '0';
  signal wdata : std_logic_vector(31 downto 0) := (others => '0');
  signal awready, wready, bvalid, arready, rvalid : std_logic;
  signal rdata : std_logic_vector(31 downto 0);

  signal net_start, head_start : std_logic;
  signal net_triggers, head_triggers : std_logic_vector(2 downto 0);
  signal net_idle, head_idle : std_logic := '1';
  signal net_done, head_done : std_logic := '0';
  signal net_return, head_return : std_logic_vector(7 downto 0) := (others => '0');
  signal net_decouple, head_decouple : std_logic := '0';
  signal net_resetn, head_resetn : std_logic := '1';
  signal fm_write_addr : std_logic_vector(31 downto 0) := (others => '0');
  signal fm_write_en : std_logic := '0';
  signal fm_write_we : std_logic_vector(1 downto 0) := "00";
  signal fm_write_din : std_logic_vector(15 downto 0) := (others => '0');
  signal fm_write_dout : std_logic_vector(15 downto 0);
  signal fm_read_addr : std_logic_vector(31 downto 0) := (others => '0');
  signal fm_read_en : std_logic := '0';
  signal fm_read_dout : std_logic_vector(15 downto 0);

  -- what the models hold and answer
  signal net_module, head_module : integer := 0;           -- after boot: CNN and circle head
  signal net_loads, head_loads : integer := 0;
  signal selector_excludes, head_answer, selector_top1, selector_confident : integer := 0;
  signal checksum_errors : integer := 0;
  signal finished : boolean := false;
begin
  clk <= not clk after 5 ns when not finished else clk;

  dut : entity work.cascade_controller
    port map (
      clk => clk, resetn => resetn,
      s_axi_awaddr => awaddr, s_axi_awvalid => awvalid, s_axi_awready => awready, s_axi_wdata => wdata, s_axi_wstrb => "1111",
      s_axi_wvalid => wvalid, s_axi_wready => wready, s_axi_bresp => open, s_axi_bvalid => bvalid, s_axi_bready => '1',
      s_axi_araddr => araddr, s_axi_arvalid => arvalid, s_axi_arready => arready, s_axi_rdata => rdata, s_axi_rresp => open,
      s_axi_rvalid => rvalid, s_axi_rready => '1',
      net_ap_start => net_start, net_ap_idle => net_idle, net_ap_done => net_done, net_ap_return => net_return,
      net_decouple => net_decouple, net_resetn => net_resetn, net_triggers => net_triggers,
      head_ap_start => head_start, head_ap_idle => head_idle, head_ap_done => head_done, head_ap_return => head_return,
      head_decouple => head_decouple, head_resetn => head_resetn, head_triggers => head_triggers,
      fm_write_addr => fm_write_addr, fm_write_en => fm_write_en, fm_write_we => fm_write_we, fm_write_din => fm_write_din,
      fm_write_dout => fm_write_dout, fm_write_clk => clk, fm_write_rst => '0',
      fm_read_addr => fm_read_addr, fm_read_en => fm_read_en, fm_read_we => "00", fm_read_din => x"0000",
      fm_read_dout => fm_read_dout, fm_read_clk => clk, fm_read_rst => '0');

  -- model of the DFX Controller, network slot: a hardware trigger loads a module (decouple, load time, reset)
  net_dfx : process
  begin
    wait until rising_edge(clk) and net_triggers /= "000" and net_decouple = '0';
    net_decouple <= '1'; net_resetn <= '0'; net_loads <= net_loads + 1;
    if net_triggers = "001" then net_module <= 0; elsif net_triggers = "010" then net_module <= 1; else net_module <= 2; end if;
    for i in 1 to 300 loop wait until rising_edge(clk); end loop;
    net_decouple <= '0';
    for i in 1 to 8 loop wait until rising_edge(clk); end loop;
    net_resetn <= '1';
  end process net_dfx;

  head_dfx : process
  begin
    wait until rising_edge(clk) and head_triggers /= "000" and head_decouple = '0';
    head_decouple <= '1'; head_resetn <= '0'; head_loads <= head_loads + 1;
    if head_triggers = "001" then head_module <= 0; elsif head_triggers = "010" then head_module <= 1; else head_module <= 2; end if;
    for i in 1 to 250 loop wait until rising_edge(clk); end loop;
    head_decouple <= '0';
    for i in 1 to 8 loop wait until rising_edge(clk); end loop;
    head_resetn <= '1';
  end process head_dfx;

  -- model of the network slot: CNN answers class 1, SNN class 2, the selector writes the feature map (640 values) through the BRAM port
  network : process
  begin
    wait until rising_edge(clk) and net_start = '1' and net_idle = '1' and net_resetn = '1';
    net_idle <= '0';
    for i in 1 to 50 loop wait until rising_edge(clk); end loop;
    if net_module = 2 then
      for i in 0 to 639 loop
        fm_write_addr <= std_logic_vector(to_unsigned(i * 2, 32)); fm_write_din <= std_logic_vector(to_unsigned(i + 256, 16));
        fm_write_we <= "11"; fm_write_en <= '1';
        wait until rising_edge(clk);
      end loop;
      fm_write_en <= '0'; fm_write_we <= "00";
      net_return <= std_logic_vector(to_unsigned(selector_excludes + selector_top1 * 4 + selector_confident * 16, 8));
    elsif net_module = 0 then
      net_return <= x"01";
    else
      net_return <= x"02";
    end if;
    for i in 1 to 20 loop wait until rising_edge(clk); end loop;
    net_done <= '1'; wait until rising_edge(clk); net_done <= '0'; net_idle <= '1';
  end process network;

  -- model of the head slot: reads the feature map through the BRAM port (data one clock after the address) and answers
  head : process
    variable expected : unsigned(15 downto 0);
  begin
    wait until rising_edge(clk) and head_start = '1' and head_idle = '1' and head_resetn = '1';
    head_idle <= '0';
    for i in 1 to 10 loop wait until rising_edge(clk); end loop;
    for j in 0 to 639 loop
      fm_read_addr <= std_logic_vector(to_unsigned(j * 2, 32)); fm_read_en <= '1';
      wait until rising_edge(clk);
      wait for 1 ns;                                               -- the data of address j is on fm_read_dout one clock later
      if unsigned(fm_read_dout) /= to_unsigned(j + 256, 16) then checksum_errors <= checksum_errors + 1; end if;
    end loop;
    fm_read_en <= '0';
    head_return <= std_logic_vector(to_unsigned(head_answer, 8));
    for i in 1 to 5 loop wait until rising_edge(clk); end loop;
    head_done <= '1'; wait until rising_edge(clk); head_done <= '0'; head_idle <= '1';
  end process head;

  -- the PS: AXI-Lite accesses and the test sequence
  stimulus : process
    variable errors : integer := 0;
    variable value : std_logic_vector(31 downto 0);

    procedure axi_write(address : in std_logic_vector(7 downto 0); data : in std_logic_vector(31 downto 0)) is
    begin
      wait until rising_edge(clk);
      awaddr <= address; wdata <= data; awvalid <= '1'; wvalid <= '1';
      wait until rising_edge(clk);
      while not (awready = '1' and wready = '1') loop wait until rising_edge(clk); end loop;
      awvalid <= '0'; wvalid <= '0';
    end procedure;

    procedure axi_read(address : in std_logic_vector(7 downto 0); data : out std_logic_vector(31 downto 0)) is
    begin
      wait until rising_edge(clk);
      araddr <= address; arvalid <= '1';
      wait until rising_edge(clk);
      while arready = '0' loop wait until rising_edge(clk); end loop;
      arvalid <= '0';
      while rvalid = '0' loop wait until rising_edge(clk); end loop;
      data := rdata;
      wait until rising_edge(clk);
    end procedure;

    -- run(mode, class the selector excludes, head answer, the selector's top-1 class, selector confident (0/1), expected final class, expected network slot module, expected head slot module,
    --     smallest and largest expected number of reconfiguration cycles)
    procedure run(mode, exclude, answer, top1, confident, expected_class, expected_net, expected_head, min_reconfig, max_reconfig : in integer) is
      variable status, result, net_load, net_run, head_load, head_run, total : std_logic_vector(31 downto 0);
      variable reconfig : integer;
      variable polls : integer := 0;
      variable checksum_errors_before : integer;
    begin
      checksum_errors_before := checksum_errors;
      selector_excludes <= exclude; head_answer <= answer; selector_top1 <= top1; selector_confident <= confident;
      wait until rising_edge(clk);
      axi_write(x"00", std_logic_vector(to_unsigned(mode * 2 + 1, 32)));
      status := (others => '0');
      while status(1) = '0' and polls < 2000000 loop axi_read(x"04", status); polls := polls + 1; end loop;
      axi_read(x"08", result); axi_read(x"0C", net_load); axi_read(x"10", net_run); axi_read(x"14", head_load); axi_read(x"18", head_run); axi_read(x"1C", total);
      reconfig := to_integer(unsigned(net_load)) + to_integer(unsigned(head_load));
      report "mode " & integer'image(mode) & " exclude " & integer'image(exclude) & " answer " & integer'image(answer) &
             " -> class " & integer'image(to_integer(unsigned(result(1 downto 0)))) & " excluded " & integer'image(to_integer(unsigned(result(9 downto 8)))) &
             " reconfig " & integer'image(reconfig) & " run " & integer'image(to_integer(unsigned(net_run)) + to_integer(unsigned(head_run))) & " total " & integer'image(to_integer(unsigned(total))) &
             " (net " & integer'image(net_module) & " head " & integer'image(head_module) & ")";
      if status(2) = '1' then errors := errors + 1; report "FAIL: error flag" severity error; end if;
      if status(1) = '0' or status(0) = '1' then errors := errors + 1; report "FAIL: done/busy" severity error; end if;
      if to_integer(unsigned(result(1 downto 0))) /= expected_class then errors := errors + 1; report "FAIL: final class" severity error; end if;
      if to_integer(unsigned(status(9 downto 8))) /= expected_net or net_module /= expected_net then errors := errors + 1; report "FAIL: network slot module" severity error; end if;
      if to_integer(unsigned(status(11 downto 10))) /= expected_head or head_module /= expected_head then errors := errors + 1; report "FAIL: head slot module" severity error; end if;
      if reconfig < min_reconfig or reconfig > max_reconfig then errors := errors + 1; report "FAIL: reconfig cycles" severity error; end if;
      if (mode = 2 and confident = 0) /= (result(17) = '1') then errors := errors + 1; report "FAIL: head ran flag" severity error; end if;
      if to_integer(unsigned(total)) < to_integer(unsigned(net_load)) + to_integer(unsigned(net_run)) + to_integer(unsigned(head_load)) + to_integer(unsigned(head_run)) then errors := errors + 1; report "FAIL: total cycles" severity error; end if;
      if mode = 2 and confident = 1 and result(16) /= '1' then errors := errors + 1; report "FAIL: head not reported as skipped" severity error; end if;
      if not (mode = 2 and confident = 1) and result(16) /= '0' then errors := errors + 1; report "FAIL: head reported as skipped" severity error; end if;
      if mode = 2 and confident = 0 and checksum_errors /= checksum_errors_before then errors := errors + 1; report "FAIL: the head read a wrong feature map" severity error; end if;
    end procedure;
  begin
    for i in 1 to 10 loop wait until rising_edge(clk); end loop;
    resetn <= '1';
    for i in 1 to 10 loop wait until rising_edge(clk); end loop;
    -- from boot the slots hold the CNN and the circle head
    run(0, 0, 0, 0, 0, 1, 0, 0,   0,   0);        -- CNN: nothing to load
    run(1, 0, 0, 0, 0, 2, 1, 0, 250, 450);        -- SNN: network slot load
    run(2, 0, 0, 0, 0, 1, 2, 0, 250, 450);        -- cascade, excluded 0: selector load; head 0 (circle) is already there. Pair {1,2}, answer 0 -> class 1
    run(2, 0, 1, 0, 0, 2, 2, 0,   0,   0);        -- same again, answer 1 -> class 2: nothing loaded
    run(2, 1, 0, 0, 0, 0, 2, 1, 250, 450);        -- excluded 1: head 1 loaded. Pair {0,2}, answer 0 -> class 0
    run(2, 1, 1, 0, 0, 2, 2, 1,   0,   0);        -- answer 1 -> class 2
    run(2, 2, 0, 0, 0, 0, 2, 2, 250, 450);        -- excluded 2: head 2 loaded. Pair {0,1}, answer 0 -> class 0
    run(2, 2, 1, 0, 0, 1, 2, 2,   0,   0);        -- answer 1 -> class 1
    run(2, 0, 0, 0, 0, 1, 2, 0, 250, 450);        -- excluded 0 again: head 0 comes back (head slot load only)
    -- confident selector: its top-1 is the answer, the head is neither loaded nor run (the head slot keeps head 0)
    run(2, 1, 0, 2, 1, 2, 2, 0,   0,   0);        -- excluded 1 would need head 1, but the selector is confident of class 2
    run(2, 2, 1, 0, 1, 0, 2, 0,   0,   0);        -- excluded 2 would need head 2: confident of class 0
    run(2, 1, 1, 1, 0, 2, 2, 1, 250, 450);        -- not confident again: head 1 is loaded and decides (answer 1 -> class 2)
    run(0, 0, 0, 0, 0, 1, 0, 1, 250, 450);        -- CNN after the cascade: network slot load
    if errors = 0 then
      report "ALL TESTS PASSED (network loads " & integer'image(net_loads) & ", head loads " & integer'image(head_loads) & ")";
    else
      report "TESTS FAILED: " & integer'image(errors) & " errors" severity failure;
    end if;
    finished <= true;
    wait;
  end process stimulus;
end architecture sim;
