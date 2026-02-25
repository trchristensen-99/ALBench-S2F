import sys

import pexpect


def scp_file(src, dest):
    print(f"Transferring {src} to {dest}...")
    # The SCP command through the SSH proxy jump
    cmd = f"scp -o ProxyJump=christen@ssh.cshl.edu -i ~/.ssh/id_ed25519_citra {src} {dest}"

    # Spawn the process
    child = pexpect.spawn(cmd, encoding="utf-8", timeout=60)

    # Expect sequence
    # 1. First password (for ssh.cshl.edu)
    child.expect("password:")
    child.sendline("R!ghth0us3")

    # 2. First Duo prompt (for ssh.cshl.edu)
    child.expect("Passcode or option")
    child.sendline("1")

    # 3. Second password (for steinitz.cshl.edu/bamdev4)
    child.expect("password:")
    child.sendline("R!ghth0us3")

    # 4. Second Duo prompt
    child.expect("Passcode or option")
    child.sendline("1")

    # Wait for completion
    child.expect(pexpect.EOF)
    print(child.before)
    print(f"Successfully transferred {src}!")


if __name__ == "__main__":
    src1 = "/Users/christen/Downloads/ALBench-S2F/models/alphagenome_heads.py"
    dest1 = "christen@bamdev4.cshl.edu:/grid/wsbs/home_norepl/christen/ALBench-S2F/models/alphagenome_heads.py"

    src2 = "/Users/christen/Downloads/ALBench-S2F/experiments/train_oracle_alphagenome.py"
    dest2 = "christen@bamdev4.cshl.edu:/grid/wsbs/home_norepl/christen/ALBench-S2F/experiments/train_oracle_alphagenome.py"

    try:
        scp_file(src1, dest1)
        scp_file(src2, dest2)
    except pexpect.exceptions.TIMEOUT as e:
        print("Timeout occurred during SCP transfer.")
        print(str(e))
        sys.exit(1)
    except pexpect.exceptions.EOF as e:
        print("Unexpected EOF occurred during SCP transfer.")
        print(str(e))
        sys.exit(1)
