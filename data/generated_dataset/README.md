Data array structure
====================

Structure of generated data:

[[Variables],[Prices],[Genome],[Load_Profile_El.],[Feasibility],[Distance, Repaired_Schedule]]

    0: Variables
        1. Consumption el. key: Some models may set this to "None" representing an array of 0's
        2. Consumption heat key
        3. Has CHP
        4. CHP initial mode
        5. CHP initial staying time
        6. CHP initial storage charge (in %)
        7. Has battery
        8. Battery initial charge (in %)

    1: Prices (Approach: indirect)

    2: Genome (Approach: generation)

    3: Load profile el.

    4: Feasibility (Approach: classification)

    5: Distance, Repaired schedule (Approach: validation and repair)
