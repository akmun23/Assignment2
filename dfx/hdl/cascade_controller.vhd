----------------------------------------------------------------------------------
-- cascade_controller: Runs the networks and the cascade without the PS: loads them through the DFX Controller, starts them and holds the feature-map RAM
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity cascade_controller is
  port (
    clk    : in  std_logic;
    resetn : in  std_logic;

    -- PS: AXI4-Lite slave
    s_axi_awaddr  : in  std_logic_vector(7 downto 0);
    s_axi_awvalid : in  std_logic;
    s_axi_awready : out std_logic;
    s_axi_wdata   : in  std_logic_vector(31 downto 0);
    s_axi_wstrb   : in  std_logic_vector(3 downto 0);
    s_axi_wvalid  : in  std_logic;
    s_axi_wready  : out std_logic;
    s_axi_bresp   : out std_logic_vector(1 downto 0);
    s_axi_bvalid  : out std_logic;
    s_axi_bready  : in  std_logic;
    s_axi_araddr  : in  std_logic_vector(7 downto 0);
    s_axi_arvalid : in  std_logic;
    s_axi_arready : out std_logic;
    s_axi_rdata   : out std_logic_vector(31 downto 0);
    s_axi_rresp   : out std_logic_vector(1 downto 0);
    s_axi_rvalid  : out std_logic;
    s_axi_rready  : in  std_logic;

    -- network slot (CNN, SNN or selector)
    net_ap_start  : out std_logic;
    net_ap_idle   : in  std_logic;
    net_ap_done   : in  std_logic;
    net_ap_return : in  std_logic_vector(7 downto 0);   -- class result: CNN, SNN: highest score. Selector: bits 1:0 lowest score = excluded class, bits 3:2 highest score, bit 4 confident
    net_decouple  : in  std_logic;                       -- DFX Controller: the slot is being reconfigured
    net_resetn    : in  std_logic;                       -- reset of the module in the slot (low during and just after a load)
    net_triggers  : out std_logic_vector(2 downto 0);    -- DFX Controller hardware triggers: trigger i loads module i

    -- head slot (circle, square or triangle head)
    head_ap_start  : out std_logic;
    head_ap_idle   : in  std_logic;
    head_ap_done   : in  std_logic;
    head_ap_return : in  std_logic_vector(7 downto 0);  -- class result: 0 = lower-numbered class of the pair, 1 = higher-numbered one
    head_decouple  : in  std_logic;
    head_resetn    : in  std_logic;
    head_triggers  : out std_logic_vector(2 downto 0);

    -- feature-map RAM written by the selector (BRAM interface, this block is the memory)
    fm_write_addr : in  std_logic_vector(31 downto 0);   -- byte address
    fm_write_en   : in  std_logic;
    fm_write_we   : in  std_logic_vector(1 downto 0);
    fm_write_din  : in  std_logic_vector(15 downto 0);
    fm_write_dout : out std_logic_vector(15 downto 0);
    fm_write_clk  : in  std_logic;
    fm_write_rst  : in  std_logic;

    -- feature-map RAM read by the head
    fm_read_addr : in  std_logic_vector(31 downto 0);    -- byte address
    fm_read_en   : in  std_logic;
    fm_read_we   : in  std_logic_vector(1 downto 0);     -- unused: the head only reads (the port exists because the HLS bram interface has it)
    fm_read_din  : in  std_logic_vector(15 downto 0);    -- unused
    fm_read_dout : out std_logic_vector(15 downto 0);
    fm_read_clk  : in  std_logic;
    fm_read_rst  : in  std_logic
  );

end entity cascade_controller;

architecture Behavioral of cascade_controller is
  -- Vivado infers the buses from these attributes (module reference in a block design).
  attribute X_INTERFACE_INFO      : string;
  attribute X_INTERFACE_MODE      : string;
  attribute X_INTERFACE_PARAMETER : string;
  attribute X_INTERFACE_INFO of clk    : signal is "xilinx.com:signal:clock:1.0 clk CLK";
  attribute X_INTERFACE_PARAMETER of clk : signal is "ASSOCIATED_BUSIF s_axi:fm_write:fm_read, ASSOCIATED_RESET resetn";
  attribute X_INTERFACE_INFO of resetn : signal is "xilinx.com:signal:reset:1.0 resetn RST";
  attribute X_INTERFACE_PARAMETER of resetn : signal is "POLARITY ACTIVE_LOW";
  attribute X_INTERFACE_INFO of fm_write_addr : signal is "xilinx.com:interface:bram:1.0 fm_write ADDR";
  attribute X_INTERFACE_MODE of fm_write_addr : signal is "slave";
  attribute X_INTERFACE_PARAMETER of fm_write_addr : signal is "MASTER_TYPE BRAM_CTRL, MEM_ADDRESS_MODE BYTE_ADDRESS, MEM_ECC NONE, MEM_SIZE 1280, MEM_WIDTH 16, READ_LATENCY 1";
  attribute X_INTERFACE_INFO of fm_write_en   : signal is "xilinx.com:interface:bram:1.0 fm_write EN";
  attribute X_INTERFACE_INFO of fm_write_we   : signal is "xilinx.com:interface:bram:1.0 fm_write WE";
  attribute X_INTERFACE_INFO of fm_write_din  : signal is "xilinx.com:interface:bram:1.0 fm_write DIN";
  attribute X_INTERFACE_INFO of fm_write_dout : signal is "xilinx.com:interface:bram:1.0 fm_write DOUT";
  attribute X_INTERFACE_INFO of fm_write_clk  : signal is "xilinx.com:interface:bram:1.0 fm_write CLK";
  attribute X_INTERFACE_INFO of fm_write_rst  : signal is "xilinx.com:interface:bram:1.0 fm_write RST";
  attribute X_INTERFACE_INFO of fm_read_addr  : signal is "xilinx.com:interface:bram:1.0 fm_read ADDR";
  attribute X_INTERFACE_MODE of fm_read_addr  : signal is "slave";
  attribute X_INTERFACE_PARAMETER of fm_read_addr : signal is "MASTER_TYPE BRAM_CTRL, MEM_ADDRESS_MODE BYTE_ADDRESS, MEM_ECC NONE, MEM_SIZE 1280, MEM_WIDTH 16, READ_LATENCY 1";
  attribute X_INTERFACE_INFO of fm_read_en    : signal is "xilinx.com:interface:bram:1.0 fm_read EN";
  attribute X_INTERFACE_INFO of fm_read_we    : signal is "xilinx.com:interface:bram:1.0 fm_read WE";
  attribute X_INTERFACE_INFO of fm_read_din   : signal is "xilinx.com:interface:bram:1.0 fm_read DIN";
  attribute X_INTERFACE_INFO of fm_read_dout  : signal is "xilinx.com:interface:bram:1.0 fm_read DOUT";
  attribute X_INTERFACE_INFO of fm_read_clk   : signal is "xilinx.com:interface:bram:1.0 fm_read CLK";
  attribute X_INTERFACE_INFO of fm_read_rst   : signal is "xilinx.com:interface:bram:1.0 fm_read RST";
  constant MODE_CNN     : unsigned(1 downto 0) := "00";
  constant MODE_SNN     : unsigned(1 downto 0) := "01";
  constant MODE_CASCADE : unsigned(1 downto 0) := "10";
  constant MODULE_SELECTOR : unsigned(1 downto 0) := "10";    -- module of the network slot that runs the first half of the cascade

  type STATE_TYPE is (
    IDLE,            -- waiting for CONTROL
    START_NETWORK,   -- start the network slot (one clock)
    WAIT_NETWORK,    -- the network slot loads and runs its module
    DECIDE,          -- look at the network's result: finished, or a head is needed
    START_HEAD,      -- start the head slot (one clock)
    WAIT_HEAD,       -- the head slot loads and runs its module
    FINISH
  );
  -- Declare current and next state signals
  signal current_state : STATE_TYPE := IDLE;
  signal next_state    : STATE_TYPE := IDLE;

  signal mode          : unsigned(1 downto 0) := MODE_CNN;
  signal start_request : std_logic := '0';                  -- from the AXI process: CONTROL was written with start
  signal start_mode    : unsigned(1 downto 0) := MODE_CNN;

  -- the two slots
  signal net_start, net_done, net_failed, net_loading, net_running : std_logic;
  signal head_start, head_done, head_failed, head_loading, head_running : std_logic;
  signal net_wanted, head_wanted, net_loaded, head_loaded : unsigned(1 downto 0);
  signal net_result, head_result : std_logic_vector(7 downto 0);

  -- the selector's result: bits 1:0 class with the lowest score (excluded), bits 3:2 class with the highest score, bit 4 confident
  signal excluded_class, top1_class : unsigned(1 downto 0);
  signal confident     : std_logic;
  signal needs_head    : std_logic;                         -- cascade and the selector is not sure
  signal bad_class     : std_logic;                         -- the selector reported a class that does not exist

  signal final_class    : unsigned(1 downto 0) := "00";
  signal head_skipped   : std_logic := '0';                 -- the selector was confident, no head ran
  signal head_ran       : std_logic := '0';                 -- a head ran in this cascade
  signal busy           : std_logic;
  signal done, error    : std_logic := '0';
  -- clock cycles of the last run: loading and running in each slot, and from start to done
  signal net_load_cycles, net_run_cycles, head_load_cycles, head_run_cycles, total_cycles : unsigned(31 downto 0) := (others => '0');

  -- The two classes that remain after the selector excluded one, lower-numbered first; `upper` picks one of them.
  function remaining_class(excluded : unsigned(1 downto 0); upper : std_logic) return unsigned is
  begin
    if upper = '0' then
      if excluded = "00" then return "01"; else return "00"; end if;
    else
      if excluded = "10" then return "01"; else return "10"; end if;
    end if;
  end function remaining_class;

  -- feature-map RAM (1024 x 16 bit, one RAMB18): the selector writes it, a head reads it
  type ram_t is array (0 to 1023) of std_logic_vector(15 downto 0);
  signal feature_map_ram : ram_t := (others => (others => '0'));
  attribute ram_style : string;
  attribute ram_style of feature_map_ram : signal is "block";
  signal writes_enabled : std_logic := '0';                    -- only a running network slot may write (its outputs are undefined while it loads)
  signal fm_read_data   : std_logic_vector(15 downto 0) := (others => '0');

  -- AXI4-Lite
  signal awready, wready, bvalid, arready, rvalid : std_logic := '0';
  signal rdata : std_logic_vector(31 downto 0) := (others => '0');
begin
  -- ------------------------------------------------------------------------------------------------------------
  -- feature-map RAM
  -- ------------------------------------------------------------------------------------------------------------
  fm_write_dout <= (others => '0');
  fm_read_dout  <= fm_read_data;

  ram : process (clk)
  begin
    if rising_edge(clk) then
      if writes_enabled = '1' and fm_write_en = '1' and fm_write_we /= "00" then
        feature_map_ram(to_integer(unsigned(fm_write_addr(10 downto 1)))) <= fm_write_din;
      end if;
      if fm_read_en = '1' then
        fm_read_data <= feature_map_ram(to_integer(unsigned(fm_read_addr(10 downto 1))));   -- data one clock after the address, as the HLS bram interface expects
      end if;
    end if;
  end process ram;

  -- ------------------------------------------------------------------------------------------------------------
  -- the two slots: each loads its module if needed and runs it
  -- ------------------------------------------------------------------------------------------------------------
  net_wanted  <= MODULE_SELECTOR when mode = MODE_CASCADE else mode;      -- CNN, SNN, or the selector for the cascade
  head_wanted <= excluded_class;                                           -- the head is named after the class the selector excluded

  network_runner : entity work.slot_runner
    port map (clk => clk, resetn => resetn,
              start => net_start, wanted => net_wanted, done => net_done, failed => net_failed, result => net_result,
              loaded => net_loaded, loading => net_loading, running => net_running,
              triggers => net_triggers, decouple => net_decouple, module_resetn => net_resetn,
              ap_start => net_ap_start, ap_idle => net_ap_idle, ap_done => net_ap_done, ap_return => net_ap_return);

  head_runner : entity work.slot_runner
    port map (clk => clk, resetn => resetn,
              start => head_start, wanted => head_wanted, done => head_done, failed => head_failed, result => head_result,
              loaded => head_loaded, loading => head_loading, running => head_running,
              triggers => head_triggers, decouple => head_decouple, module_resetn => head_resetn,
              ap_start => head_ap_start, ap_idle => head_ap_idle, ap_done => head_ap_done, ap_return => head_ap_return);

  excluded_class <= unsigned(net_result(1 downto 0));
  top1_class     <= unsigned(net_result(3 downto 2));
  confident      <= net_result(4);
  bad_class      <= '1' when excluded_class = "11" or top1_class = "11" else '0';
  needs_head     <= '1' when mode = MODE_CASCADE and bad_class = '0' and confident = '0' else '0';

  net_start  <= '1' when current_state = START_NETWORK else '0';
  head_start <= '1' when current_state = START_HEAD else '0';
  busy       <= '0' when current_state = IDLE else '1';

  -- ------------------------------------------------------------------------------------------------------------
  -- cascade sequence: network slot, then (cascade only, and only if the selector is not sure) the head slot
  -- ------------------------------------------------------------------------------------------------------------
  clocked : process(clk)
  begin
    if rising_edge(clk) then
      if resetn = '0' then
        current_state   <= IDLE;
        mode            <= MODE_CNN;
        final_class     <= "00";
        head_skipped    <= '0';
        done            <= '0';
        error           <= '0';
        head_ran        <= '0';
        net_load_cycles <= (others => '0');
        net_run_cycles  <= (others => '0');
        head_load_cycles <= (others => '0');
        head_run_cycles <= (others => '0');
        total_cycles    <= (others => '0');
        writes_enabled  <= '0';
      else
        current_state  <= next_state;
        writes_enabled <= net_running;       -- only a running network slot may write the feature map (its outputs are undefined while it loads)
        if net_loading = '1' then
          net_load_cycles <= net_load_cycles + 1;
        end if;
        if net_running = '1' then
          net_run_cycles <= net_run_cycles + 1;
        end if;
        if head_loading = '1' then
          head_load_cycles <= head_load_cycles + 1;
        end if;
        if head_running = '1' then
          head_run_cycles <= head_run_cycles + 1;
        end if;
        if busy = '1' then
          total_cycles <= total_cycles + 1;
        end if;

        case current_state is
          when IDLE =>
            if start_request = '1' then
              mode            <= start_mode;
              done            <= '0';
              error           <= '0';
              head_skipped    <= '0';
              head_ran        <= '0';
              net_load_cycles <= (others => '0');
              net_run_cycles  <= (others => '0');
              head_load_cycles <= (others => '0');
              head_run_cycles <= (others => '0');
              total_cycles    <= (others => '0');
              if start_mode = "11" then      -- not a mode
                error <= '1';
              end if;
            end if;
          when WAIT_NETWORK =>
            if net_done = '1' and net_failed = '1' then
              error <= '1';
            end if;
          when DECIDE =>
            final_class <= unsigned(net_result(1 downto 0));        -- CNN, SNN: the prediction
            if mode = MODE_CASCADE and bad_class = '1' then
              error <= '1';
            elsif mode = MODE_CASCADE and confident = '1' then
              final_class  <= top1_class;                           -- the selector is sure: no head
              head_skipped <= '1';
            end if;
          when WAIT_HEAD =>
            if head_done = '1' and head_failed = '1' then
              error <= '1';
            elsif head_done = '1' then
              final_class <= remaining_class(excluded_class, head_result(0));   -- 0 = lower-numbered class of the pair, 1 = higher-numbered one
            end if;
          when FINISH =>
            done <= '1';
          when START_HEAD =>
            head_ran <= '1';
          when START_NETWORK =>
            null;
        end case;
      end if;
    end if;
  end process clocked;

  -- ------------------------------------------------------------------------------------------------------------
  -- next state
  -- ------------------------------------------------------------------------------------------------------------
  next_state_logic : process(current_state, start_request, start_mode, net_done, net_failed, needs_head, head_done)
  begin
    next_state <= current_state;

    case current_state is
      when IDLE =>
        if start_request = '1' then
          if start_mode = "11" then
            next_state <= FINISH;
          else
            next_state <= START_NETWORK;
          end if;
        end if;
      when START_NETWORK =>
        next_state <= WAIT_NETWORK;
      when WAIT_NETWORK =>
        if net_done = '1' then
          if net_failed = '1' then
            next_state <= FINISH;
          else
            next_state <= DECIDE;
          end if;
        end if;
      when DECIDE =>
        if needs_head = '1' then
          next_state <= START_HEAD;
        else
          next_state <= FINISH;
        end if;
      when START_HEAD =>
        next_state <= WAIT_HEAD;
      when WAIT_HEAD =>
        if head_done = '1' then
          next_state <= FINISH;
        end if;
      when FINISH =>
        next_state <= IDLE;
    end case;
  end process next_state_logic;

  -- ------------------------------------------------------------------------------------------------------------
  -- AXI4-Lite slave (one outstanding access, write address and data accepted together)
  -- ------------------------------------------------------------------------------------------------------------
  -- Register map (32 bit):
  --   0x00 CONTROL           write: bit 0 = start, bits 2:1 = mode (0 CNN, 1 SNN, 2 cascade); ignored while busy
  --   0x04 STATUS            bit 0 busy, bit 1 done (cleared by start), bit 2 error, bits 9:8 module in the network slot (0 CNN, 1 SNN, 2 selector),
  --                          bits 11:10 module in the head slot (0 circle, 1 square, 2 triangle)
  --   0x08 RESULT            bits 1:0 final class (0 circle, 1 square, 2 triangle), bits 9:8 class the selector excluded (names the head),
  --                          bit 16 head skipped (selector confident), bit 17 head ran
  --   0x0C NET_LOAD_CYCLES   clock cycles of the last run: network slot loading
  --   0x10 NET_RUN_CYCLES                                  network slot running
  --   0x14 HEAD_LOAD_CYCLES                                head slot loading
  --   0x18 HEAD_RUN_CYCLES                                 head slot running
  --   0x1C TOTAL_CYCLES                                    from start to done
  s_axi_awready <= awready;
  s_axi_wready  <= wready;
  s_axi_bvalid  <= bvalid;
  s_axi_bresp   <= "00";
  s_axi_arready <= arready;
  s_axi_rvalid  <= rvalid;
  s_axi_rresp   <= "00";
  s_axi_rdata   <= rdata;

  axi_lite : process (clk)
    variable word : std_logic_vector(31 downto 0);
  begin
    if rising_edge(clk) then
      if resetn = '0' then
        awready <= '0'; wready <= '0'; bvalid <= '0'; arready <= '0'; rvalid <= '0'; rdata <= (others => '0');
        start_request <= '0'; start_mode <= "00";
      else
        awready <= '0'; wready <= '0'; arready <= '0';
        start_request <= '0';                                    -- one clock pulse; the sequence takes it in IDLE

        if s_axi_awvalid = '1' and s_axi_wvalid = '1' and bvalid = '0' and awready = '0' then
          awready <= '1'; wready <= '1'; bvalid <= '1';
          if s_axi_awaddr(7 downto 2) = "000000" and s_axi_wdata(0) = '1' and busy = '0' then
            start_request <= '1'; start_mode <= unsigned(s_axi_wdata(2 downto 1));
          end if;
        end if;
        if bvalid = '1' and s_axi_bready = '1' then bvalid <= '0'; end if;

        if s_axi_arvalid = '1' and rvalid = '0' and arready = '0' then
          arready <= '1'; rvalid <= '1';
          word := (others => '0');
          case s_axi_araddr(7 downto 2) is
            when "000000" => word(2 downto 1) := std_logic_vector(mode);
            when "000001" => word(0) := busy; word(1) := done; word(2) := error;
                             word(9 downto 8) := std_logic_vector(net_loaded); word(11 downto 10) := std_logic_vector(head_loaded);
            when "000010" => word(1 downto 0) := std_logic_vector(final_class); word(9 downto 8) := std_logic_vector(excluded_class); word(16) := head_skipped; word(17) := head_ran;
            when "000011" => word := std_logic_vector(net_load_cycles);
            when "000100" => word := std_logic_vector(net_run_cycles);
            when "000101" => word := std_logic_vector(head_load_cycles);
            when "000110" => word := std_logic_vector(head_run_cycles);
            when "000111" => word := std_logic_vector(total_cycles);
            when others   => null;
          end case;
          rdata <= word;
        end if;
        if rvalid = '1' and s_axi_rready = '1' then rvalid <= '0'; end if;
      end if;
    end if;
  end process axi_lite;
end architecture Behavioral;
