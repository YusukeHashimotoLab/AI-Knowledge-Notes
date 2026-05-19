/**
 * 第一回 研究へのAI活用勉強会 アンケートフォーム自動生成スクリプト
 *
 * 使い方:
 *   1. https://script.google.com にアクセス
 *   2. 新しいプロジェクトを作成
 *   3. このスクリプトを貼り付けて createSurveyForm() を実行
 *   4. Google認証を許可
 *   5. ログに表示されるフォームURLを取得
 */

function createSurveyForm() {
  // フォーム作成
  var form = FormApp.create('第一回 研究へのAI活用勉強会 アンケート');
  form.setDescription(
    '本日は「第一回 研究へのAI活用勉強会」にご参加いただき、ありがとうございました。\n' +
    '今後の改善と次回開催の参考にさせていただきますので、アンケートへのご回答をお願いいたします（所要時間：約3分）。\n' +
    '回答は匿名で処理し、個人が特定されることはありません。'
  );
  form.setIsQuiz(false);
  form.setCollectEmail(false);
  form.setAllowResponseEdits(false);
  form.setLimitOneResponsePerUser(false);

  // ============================================================
  // セクション1: 参加者情報
  // ============================================================
  form.addSectionHeaderItem()
    .setTitle('参加者情報');

  // Q1: 所属（プルダウン・必須）
  form.addListItem()
    .setTitle('Q1. 所属')
    .setChoiceValues([
      '学際科学フロンティア研究所',
      '大学院工学研究科',
      '大学院理学研究科',
      '大学院情報科学研究科',
      '大学院環境科学研究科',
      '大学院医学系研究科',
      '大学院生命科学研究科',
      '大学院農学研究科',
      'その他'
    ])
    .setRequired(true);

  // Q2: 職位・学年（プルダウン・必須）
  form.addListItem()
    .setTitle('Q2. 職位・学年')
    .setChoiceValues([
      '教授・准教授',
      '講師・助教',
      '研究員・ポスドク',
      '博士課程',
      '修士課程',
      '学部生',
      'その他'
    ])
    .setRequired(true);

  // ============================================================
  // セクション2: 全体評価
  // ============================================================
  form.addSectionHeaderItem()
    .setTitle('全体評価');

  // Q3: 満足度（均等目盛り 1-5・必須）
  form.addScaleItem()
    .setTitle('Q3. 勉強会全体の満足度を教えてください')
    .setBounds(1, 5)
    .setLabels('不満', '大変満足')
    .setRequired(true);

  // Q4: 有用性（均等目盛り 1-5・必須）
  form.addScaleItem()
    .setTitle('Q4. 勉強会の内容は研究活動に役立ちそうですか？')
    .setBounds(1, 5)
    .setLabels('役立たない', '大変役立つ')
    .setRequired(true);

  // ============================================================
  // セクション3: 各プログラムの評価
  // ============================================================
  form.addSectionHeaderItem()
    .setTitle('各プログラムの評価');

  // Q5: 坂口教授の講演（均等目盛り 1-5・必須）
  form.addScaleItem()
    .setTitle('Q5. 坂口教授の講演はいかがでしたか？')
    .setBounds(1, 5)
    .setLabels('物足りなかった', '大変良かった')
    .setRequired(true);

  // Q6: 西准教授の講演（均等目盛り 1-5・必須）
  form.addScaleItem()
    .setTitle('Q6. 西准教授の講演はいかがでしたか？')
    .setBounds(1, 5)
    .setLabels('物足りなかった', '大変良かった')
    .setRequired(true);

  // Q7: 1分プレゼン（均等目盛り 1-5・必須）
  form.addScaleItem()
    .setTitle('Q7. 1分プレゼンはいかがでしたか？')
    .setBounds(1, 5)
    .setLabels('物足りなかった', '大変良かった')
    .setRequired(true);

  // Q8: 交流会（均等目盛り 1-5・必須）
  form.addScaleItem()
    .setTitle('Q8. 交流会はいかがでしたか？')
    .setBounds(1, 5)
    .setLabels('物足りなかった', '大変良かった')
    .setRequired(true);

  // ============================================================
  // セクション4: AI活用状況
  // ============================================================
  form.addSectionHeaderItem()
    .setTitle('AI活用状況');

  // Q9: 利用頻度（ラジオボタン・必須）
  form.addMultipleChoiceItem()
    .setTitle('Q9. 現在、研究活動で生成AIをどの程度活用していますか？')
    .setChoiceValues([
      'ほぼ毎日使っている',
      '週に数回使っている',
      '月に数回使っている',
      'ほとんど使っていない',
      'まだ使ったことがない'
    ])
    .setRequired(true);

  // Q10: 使用ツール（チェックボックス・任意）
  form.addCheckboxItem()
    .setTitle('Q10. 主にどのAIツールを使っていますか？（複数選択可）')
    .setChoiceValues([
      'ChatGPT (OpenAI)',
      'Claude (Anthropic)',
      'Gemini (Google)',
      'GitHub Copilot',
      'NotebookLM',
      'その他'
    ]);

  // ============================================================
  // セクション5: 今後に向けて
  // ============================================================
  form.addSectionHeaderItem()
    .setTitle('今後に向けて');

  // Q11: 希望テーマ（チェックボックス・任意）
  form.addCheckboxItem()
    .setTitle('Q11. 次回の勉強会で取り上げてほしいテーマはありますか？（複数選択可）')
    .setChoiceValues([
      '論文執筆・校正へのAI活用',
      'プログラミング・コーディング支援',
      'データ分析・可視化へのAI活用',
      '画像生成・図表作成',
      '文献調査の効率化',
      '実験計画・材料探索へのAI活用',
      'AIエージェントの活用',
      'RAG（検索拡張生成）の構築と活用',
      'その他'
    ]);

  // Q12: 次回参加意向（ラジオボタン・必須）
  form.addMultipleChoiceItem()
    .setTitle('Q12. 次回も参加したいですか？')
    .setChoiceValues([
      'ぜひ参加したい',
      '内容次第で参加したい',
      'あまり参加したくない',
      '参加したくない'
    ])
    .setRequired(true);

  // Q13: 開催頻度（ラジオボタン・必須）
  form.addMultipleChoiceItem()
    .setTitle('Q13. 勉強会の開催頻度として望ましいものはどれですか？')
    .setChoiceValues([
      '月1回程度',
      '2ヶ月に1回程度',
      '学期に1回程度',
      '年1回程度'
    ])
    .setRequired(true);

  // Q14: 良かった点・改善点（段落テキスト・任意）
  form.addParagraphTextItem()
    .setTitle('Q14. 今日の勉強会で特に良かった点や改善点があればお聞かせください')
    .setRequired(false);

  // Q15: その他ご意見（段落テキスト・任意）
  form.addParagraphTextItem()
    .setTitle('Q15. その他、ご意見・ご要望があればお聞かせください')
    .setRequired(false);

  // ============================================================
  // 送信後メッセージ
  // ============================================================
  form.setConfirmationMessage(
    'ご回答ありがとうございました。いただいたご意見は今後の勉強会の改善に活用させていただきます。'
  );

  // フォームURLをログ出力
  Logger.log('フォームが作成されました！');
  Logger.log('編集用URL: ' + form.getEditUrl());
  Logger.log('回答用URL: ' + form.getPublishedUrl());
}
