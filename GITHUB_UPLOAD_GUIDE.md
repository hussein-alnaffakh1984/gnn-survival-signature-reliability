# دليل رفع المستودع على GitHub (خطوة بخطوة)

## ملاحظة مهمة قبل البدء — تجنّب الاستلال
هذا المستودع يحتوي **الكود فقط** — لا يحتوي نص الورقة (PDF/DOCX) عمداً.
السبب: أداة فحص الاستلال في المجلة (مثل iThenticate) ستطابق ورقتك مع أي
نص منشور. إذا رفعت الورقة على GitHub، ستظهر "مطابقة" لنفسها = مشكلة.
لذلك: **لا ترفع ملف الورقة أبداً على المستودع العام.**

═══════════════════════════════════════════════════════
## الطريقة 1: عبر موقع GitHub (الأسهل — بدون أوامر)
═══════════════════════════════════════════════════════

1. أنشئ حساباً على https://github.com (إن لم يكن لديك)

2. اضغط "+" أعلى اليمين → "New repository"

3. املأ:
   - Repository name:  gnn-survival-signature-reliability
   - Description:      Unified GNN framework for network reliability and
                       importance measures via the survival signature
   - اختر: Public (المجلات تطلب أن يكون عاماً)
   - لا تضع علامة على "Add README" (عندنا واحد جاهز)

4. اضغط "Create repository"

5. في الصفحة التالية، اضغط "uploading an existing file"

6. اسحب كل محتويات هذا المجلد (src/, scripts/, examples/, figures/,
   README.md, LICENSE, ...) إلى الصفحة

7. اكتب رسالة commit (مثلاً: "Initial release")

8. اضغط "Commit changes"

✓ انتهى — مستودعك الآن عام على:
   https://github.com/USERNAME/gnn-survival-signature-reliability

═══════════════════════════════════════════════════════
## الطريقة 2: عبر سطر الأوامر (Git)
═══════════════════════════════════════════════════════

```bash
cd gnn-survival-signature-reliability
git init
git add .
git commit -m "Initial release: code, data, and reproduction scripts"
git branch -M main
git remote add origin https://github.com/USERNAME/gnn-survival-signature-reliability.git
git push -u origin main
```

═══════════════════════════════════════════════════════
## بعد الرفع — للورقة
═══════════════════════════════════════════════════════

1. انسخ رابط المستودع:
   https://github.com/USERNAME/gnn-survival-signature-reliability

2. في الورقة، قسم "Data Availability" / "Code Availability"،
   استبدل [URL to be inserted upon acceptance] بالرابط.

3. (موصى به للمجلات) أنشئ DOI دائم عبر Zenodo:
   - اذهب https://zenodo.org → سجّل دخول بحساب GitHub
   - فعّل المستودع → أنشئ Release على GitHub → Zenodo يولّد DOI
   - ضع الـ DOI في الورقة محل [DOI to be inserted upon acceptance]

═══════════════════════════════════════════════════════
## تنبيه التعمية المزدوجة (Double-blind)
═══════════════════════════════════════════════════════
إذا كانت المجلة مراجعتها مزدوجة التعمية (double-blind):
- لا تضع رابط GitHub الحقيقي في النسخة المُرسَلة للمراجعة
- بدلاً منه اكتب: "Code will be made available upon acceptance"
- أضف الرابط فقط في النسخة النهائية بعد القبول
تحقّق من نوع المراجعة في تعليمات المجلة.
