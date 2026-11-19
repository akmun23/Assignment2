----------------------------------------------------------------------------------
-- slot_runner: Loads a module into one reconfigurable slot if needed and runs it (ap_ctrl_hs)
----------------------------------------------------------------------------------

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity slot_runner is
  port (
    clk    : in  std_logic;
    resetn : in  std_logic;

    -- from the cascade controller
    start   : in  std_logic;                       -- one clock pulse: load module `wanted` if it is not in the slot, then run it
    wanted  : in  unsigned(1 downto 0);            -- module to run, stable until done
    done    : out std_logic;                       -- one clock pulse when the run is over
    failed  : out std_logic;                       -- with done: the load or the run did not finish
    result  : out std_logic_vector(7 downto 0);    -- ap_return of the last run
    loaded  : out unsigned(1 downto 0);            -- module in the slot (after boot: module 0)
    loading : out std_logic;                       -- high while a module is loaded
    running : out std_logic;                       -- high while the module runs

    -- DFX Controller of this slot
    triggers      : out std_logic_vector(2 downto 0);   -- hardware trigger i loads module i
    decouple      : in  std_logic;                      -- the slot is being reconfigured
    module_resetn : in  std_logic;                      -- reset of the module in the slot (low during and just after a load)

    -- the module (ap_ctrl_hs)
    ap_start  : out std_logic;
    ap_idle   : in  std_logic;
    ap_done   : in  std_logic;
    ap_return : in  std_logic_vector(7 downto 0)
  );
end entity slot_runner;

architecture Behavioral of slot_runner is

  -- Architecture Declarations
  constant TIMEOUT_CYCLES : unsigned(23 downto 0) := to_unsigned(10_000_000, 24);   -- 100 ms: a load takes about 2 ms, a run under 1 ms

  type STATE_TYPE is (
    IDLE,
    REQUEST_LOAD,   -- trigger raised until the DFX Controller starts (decouple rises)
    WAIT_LOAD,      -- until the module is loaded and out of reset
    WAIT_IDLE,      -- until the module is idle
    START_MODULE,   -- ap_start high until the module leaves idle
    WAIT_DONE,      -- until ap_done
    FINISHED,
    TIMED_OUT
  );
  -- Declare current and next state signals
  signal current_state : STATE_TYPE := IDLE;
  signal next_state    : STATE_TYPE := IDLE;

  signal loaded_module : unsigned(1 downto 0) := "00";
  signal result_reg    : std_logic_vector(7 downto 0) := (others => '0');
  signal timeout       : unsigned(23 downto 0) := (others => '0');   -- clocks spent in the current state
  signal slot_ready    : std_logic;
begin
  slot_ready <= not decouple and module_resetn;
  loaded     <= loaded_module;
  result     <= result_reg;

  ----------------------------------------------------------------------------
  -- Registers: the state, the module in the slot, the result and the timeout
  ----------------------------------------------------------------------------
  clocked : process(clk)
  begin
    if rising_edge(clk) then
      if resetn = '0' then
        current_state <= IDLE;
        loaded_module <= "00";
        result_reg    <= (others => '0');
        timeout       <= (others => '0');
      else
        current_state <= next_state;
        if next_state /= current_state then
          timeout <= (others => '0');
        else
          timeout <= timeout + 1;
        end if;

        case current_state is
          when WAIT_LOAD =>
            if slot_ready = '1' then
              loaded_module <= wanted;
            end if;
          when WAIT_DONE =>
            if ap_done = '1' then
              result_reg <= ap_return;
            end if;
          when others =>
            null;
        end case;
      end if;
    end if;
  end process clocked;

  ----------------------------------------------------------------------------
  -- Next state
  ----------------------------------------------------------------------------
  next_state_logic : process(current_state, start, wanted, loaded_module, decouple, slot_ready, ap_idle, ap_done, timeout)
  begin
    next_state <= current_state;

    case current_state is
      when IDLE =>
        if start = '1' then
          if loaded_module /= wanted then
            next_state <= REQUEST_LOAD;
          else
            next_state <= WAIT_IDLE;
          end if;
        end if;
      when REQUEST_LOAD =>
        if decouple = '1' then
          next_state <= WAIT_LOAD;
        end if;
      when WAIT_LOAD =>
        if slot_ready = '1' then
          next_state <= WAIT_IDLE;
        end if;
      when WAIT_IDLE =>
        if slot_ready = '1' and ap_idle = '1' then
          next_state <= START_MODULE;
        end if;
      when START_MODULE =>
        if ap_idle = '0' then
          next_state <= WAIT_DONE;
        end if;
      when WAIT_DONE =>
        if ap_done = '1' then
          next_state <= FINISHED;
        end if;
      when FINISHED | TIMED_OUT =>
        next_state <= IDLE;
    end case;

    if current_state /= IDLE and timeout = TIMEOUT_CYCLES then
      next_state <= TIMED_OUT;
    end if;
  end process next_state_logic;

  ----------------------------------------------------------------------------
  -- Outputs
  ----------------------------------------------------------------------------
  outputs : process(current_state, wanted)
  begin
    done     <= '0';
    failed   <= '0';
    loading  <= '0';
    running  <= '0';
    triggers <= "000";
    ap_start <= '0';

    case current_state is
      when REQUEST_LOAD =>
        loading <= '1';
        triggers(to_integer(wanted)) <= '1';
      when WAIT_LOAD =>
        loading <= '1';
      when START_MODULE =>
        running  <= '1';
        ap_start <= '1';
      when WAIT_DONE =>
        running <= '1';
      when FINISHED =>
        done <= '1';
      when TIMED_OUT =>
        done   <= '1';
        failed <= '1';
      when IDLE | WAIT_IDLE =>
        null;
    end case;
  end process outputs;
end architecture Behavioral;
